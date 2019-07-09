# -*- coding: utf-8 -*-
import numpy as np
import cv2

class MojaOsoba:
    # Self za instancu atributa
    def __init__(self,i , x, y): 
        self.i = i
        self.x = x
        self.y = y    
        self.zavrseno = False         
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def azuriranjeKoordinata(self, x, y):
        self.age = 0
        self.x = x
        self.y = y
    def setZavrseno(self):
        self.zavrseno = True 

f = open("out.txt", "w+")
f.write("RA22-2014, Stefan Bugarinovic" + "\n") 
f.write("file,count" + "\n")
videoSnimci = range(1,11)
print ('Video snimci sa brojem ljudi na braon platou:')

for sledeciVideo in videoSnimci:
    
    snimak = "video" + format(sledeciVideo) + ".mp4"
    
    # Otvaram video
    kadar = cv2.VideoCapture(snimak) 
    # Sirina videa
    sirina = kadar.get(3)
    # Visina videa
    visina = kadar.get(4) 
    # Povrsina videa
    povrsinaFrejma = sirina*visina 
    # Promenljiva koju koristimo za detekciju ljudi
    oblast = povrsinaFrejma/900
    print ('Oblast:', oblast)    
    
    # Linije koje sluze za detekciju ljudi
    gornjaGranica =   int(1.1*(visina/5))
    donjaGranica = int(4.7*(visina/5))    
    x1 =  [0, gornjaGranica]; 
    x2 =  [sirina, gornjaGranica];
    gornjaLinija = np.array([x1,x2], np.int32)
    x3 =  [0, donjaGranica];
    x4 =  [sirina, donjaGranica];
    donjaLinija = np.array([x3,x4], np.int32)
    
    # Odvaja pokretnog od nepokretnog
    # Gausov algoritam segmentacije
    zadnjePrednje = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
    # np.uint8 - celi brojevi u opsegu 0-255
    kernelOp = np.ones((3,3),np.uint8)    
    kernelCl = np.ones((11,11),np.uint8)
    
    osobe = []
    brojacOsoba = 0
    
    while(kadar.isOpened()):
        ret, frame = kadar.read()
        # Primenjuje background subtraction
        maska1 = zadnjePrednje.apply(frame) 
        maska2 = zadnjePrednje.apply(frame)
        
        # Binarizacija za eliminisanje senki (gray color)
        try:        
            # Parametri metode cv2.threshold
            # 1. Izvorna slika koja treba da bude u sivim tonovima
            # 2. Vrednost praga koja se koristi za klasifikovanje vrednosti piksela
            # 3. Vrednost koja se dodeljuje ako je vrednost piksela veća ili manja od praga
            # 4. Stil praćenja (postoji 5 različitih)
            ret,imBin= cv2.threshold(maska1,200,255,cv2.THRESH_BINARY)
            # Otvaranje
            # Uklanjanje suma.           
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)          
            # Zatvaranje
            # Spajanje belih regiona.            
            mask2 =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)            
        except:
            print ('Broj ljudi na platou:',brojacOsoba)
            break     
       # cv2.findContours argumenti:
       # 1. Izvorna slika
       # 2. Način pronalaženja kontura
       # 3. Metoda konturne aproksimacije
        _, contours, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # Svaka kontura je niz koordinata graničnih tačaka objekta
        
        for konture in contours:
            #cv2.contourArea() funkcija daje povrsinu konture
            area = cv2.contourArea(konture)
            if area > oblast:
                # Funkcija cv2.moments () daje rečnik svih izračunatih vrednosti trenutka
                # Iz ovih trenutaka možete izvući korisne podatke kao što su područje, centroid...
                M = cv2.moments(konture)
                # Centar figure po X
                centarX = int(M['m10']/M['m00']) 
                # Centar figure po Y
                centarY = int(M['m01']/M['m00']) 
                #cv2.boundingRect() funkcija za nalazenje pravougaonika
                #(x, y) gornja leva koordinata pravokutnika, a (sirina, visina) njena širina i visina
                x,y,sirina,visina = cv2.boundingRect(konture) 
                nova = True               
                if centarY in range(gornjaGranica,donjaGranica):
                    for i in osobe:
                        #Objeat se nalazi u granicama platoa
                        if i.getY() > donjaGranica and i.getY() < gornjaGranica :
                             i.setZavrseno()
                        # Objekat je previse blizu vec detektovanom objektu
                        if abs(centarX-i.getX()) <= sirina and abs(centarY-i.getY()) <= visina:
                             nova = False
                             i.azuriranjeKoordinata(centarX,centarY)   
                    # Dodavanje nove osobe ako je detektovana
                    if nova == True:
                        novaOsoba = MojaOsoba(brojacOsoba,centarX,centarY)
                        osobe.append(novaOsoba)
                        brojacOsoba += 1     
 
        ispisNaEkran = 'Broj ljudi: '+ str(brojacOsoba)
        # cv2.polylines() koristi se za crtanje više linija
        # Napravi se opis linija i proslede se funkciji
        frame = cv2.polylines(frame,[gornjaLinija],False,(0,0,0),thickness=1)
        frame = cv2.polylines(frame,[donjaLinija],False,(0,0,0),thickness=1)
        # cv2.putText koristi se za stavljanje teksta na frejm
        cv2.putText(frame, ispisNaEkran ,(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame, ispisNaEkran ,(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)   
    
        cv2.imshow('Frame',cv2.resize(frame, (700,500)))
    
        if cv2.waitKey(1) & 0xff == ord('q') :    
            break    
    f.write(str(snimak) + "," + str(brojacOsoba) + " \n")
    # Zatvaranje videa
    kadar.release()
    #Brisanje
    cv2.destroyAllWindows()