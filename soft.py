from keras.models import load_model
import numpy as np
import cv2
import neur
import math

#vector
def dot(v, w):
    x, y = v
    x1, y1 = w
    return x * x1 + y * y1

def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)

def vector(b, e):
    x, y = b
    x1, y1 = e
    return x1 - x, y1 - y

def unit(v):
    x, y = v
    mag = length(v)
    return x / mag, y / mag

def distance(p0, p1):
    return length(vector(p0, p1))

def scale(v, sc):
    x, y = v
    return x * sc, y * sc

def add(v, w):
    x, y = v
    X, Y = w
    return x + X, y + Y

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return dist

#izracunavanje rastojanja izmedju lijije i broja
def pnt2line2(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return dist, (int(nearest[0]), int(nearest[1])), r

#transformisati selektovani region na sliku dimenzija 28x28
#elementi matrice image su vrednosti 0 ili 255. 
#skaliranje svih elementa matrice na opseg od 0 do 1
def process_region(region):
    region = cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)
    region = region.reshape((1, 28, 28, 1))
    region = region / 255
    return region

#pronalazenje linije na videu
def find_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #detektovanje plave linije na videu
    less_blue = np.array([100, 150, 100])
    more_blue = np.array([140, 255, 255])
    n = cv2.inRange(hsv, less_blue, more_blue)
    return n

#pronalazenje koordinata linije na videu
def find_line_coordinates(frame):
    mask = find_line(frame)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 50, None, 80, 10) 
    if lines is not None:
        for x, y, x1, y1 in lines[0]:
            line = x, y, x1, y1 #rezultat Hough transformacije koordinate linije
            return line
    else:
        return None

#pronalazenje broja na videu
def find_number(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sensibility = 100
    more_white = np.array([255, sensibility, 255])
    less_white = np.array([0, 0, 255 - sensibility])
    n = cv2.inRange(hsv, less_white, more_white)
    n = cv2.blur(n, ksize=(2, 2))
    n = cv2.morphologyEx(n, cv2.MORPH_CLOSE, kernel=np.ones((2, 2), dtype=np.uint8), iterations=1)
    return n

#vracanje indeksa maksimalne vrednosti duz ose
def recognize_number(model, roi):
    region = process_region(roi)
    return np.argmax(model.predict(region)[0])

#detektovati broj koji je dovoljno blizu liniji
def detect_number(numbers, number):
    numbers_array = []
    for i, number0 in enumerate(numbers):
        p, q = number0['center']
        p1, q1 = number['center']
        dist = math.sqrt(((p - p1) ** 2) + ((q - q1) ** 2))
        if dist < 12.7: #rastojanje na kome se detektuje broj
            numbers_array.append(i)
    return numbers_array


#analiziranje video frejm po frejm
if __name__ == '__main__':
    model = load_model('model.h5')
    results = []
    videos = range(0, 10)
    print('Video fajlovi sa sumom brojeva koji su prosli ispod linije:')
    for video in videos:
        cap = cv2.VideoCapture('video-{0}.avi'.format(video))
        frame_num = 0
        numbers = []
        sum_numbers = 0

        while True:
            _, frame = cap.read()
            if frame is not None:
                frame_num += 1
                counter = find_number(frame)
                #pronalazenje svih kontura na frejmu
                img, contours, hierarchy = cv2.findContours(counter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x, y, x1, y1 = find_line_coordinates(frame)

                for contour in contours:
                    p, q, w, h = cv2.boundingRect(contour)
                    if w >= 3 and h >= 7:
                        p1 = p - 2
                        q1 = q - 2
                        center = (p + w / 2, q + h / 2)
                        cv2.rectangle(frame, (p1, q1), (p + w, q + h), (0, 255, 0), 1)
                        roi = counter[q1: q + h, p1: p + w]
                        dist, pt, r = pnt2line2((p + w, q + h), (x, y), (x1, y1))
                        if dist < 9 and r == 1:
                            number = {
                                'center': center,
                                'frame_num': frame_num,
                                'history': []
                            }
                            detected_number = detect_number(numbers, number)
                            #prepoznat broj
                            if len(detected_number) == 0:
                                number['prediction'] = recognize_number(model, roi)
                                number['passed'] = False
                                numbers.append(number)
                                sum_numbers += number['prediction']
                            #broj nije prepoznat
                            elif len(detected_number) == 1:
                                idx = detected_number[0]
                                history = {
                                    'frame_num': frame_num,
                                    'center': number['center']
                                }
                                numbers[idx]['history'].append(history)
                                numbers[idx]['frame_num'] = frame_num
                                numbers[idx]['center'] = number['center']

                cv2.imshow('Video broj %d'% video, frame)
                if cv2.waitKey(1) & 0xFF == ord('b'):
                    break
            else:
                break
            
        print('Video broj %d: %d' % (video, sum_numbers))
        results.append('video-{0}.avi\t{1}\n'.format(video, sum_numbers))
        cap.release()
        cv2.destroyAllWindows()
    with open('out.txt', 'w') as file:
        file.write('Stefan Bugarinovic RA22-2014\nfile\tsum\n')
        for res in results:
            file.write(res)
    file.close()
    