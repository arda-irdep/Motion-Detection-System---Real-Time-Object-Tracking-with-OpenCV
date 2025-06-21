import cv2
import numpy as np

ogrenci_numarasi = "21yobi1049"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# Sabit olan arkaplandan hareketli objeleri çıkartmayı sağlar BackgroundSubtractorKNN class'ı da kullanılabilir
fgbg = cv2.createBackgroundSubtractorMOG2()

prev_center = None  # Algılanan objenin hızını ve yönünü hesaplamak amacıyla bir önceki merkez noktasını kullanacağız.
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Görüntünün fps değerini alır ve tamsayıya yuvarlar ondan sonra fps değişkenine atar

# Her bir frame'i (kareyi) kameradan okur eğer okunamıyorsa veya başka bir sorun oluşmuş ise döngüden çıkar
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame alınamadı!")
        break

    # Okuduğu her bir frame'i grayscale (gri seviyeli) kareye dönüştürür, bu işlem hareketin daha kolay algılanmasını sağlar.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    fgmask = fgbg.apply(gray)

    # Dış hatları (konturları) bulur. Konturlar, hareketli nesnelerin sınırlarını temsil eder.
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    speed = 0  # Piksel/saniye cinsinden hızı hesaplamak için hız değişkenini sıfıra eşitliyoruz
    angle = 0  # Derece cinsinden yönü hesaplamak için açı değişkenini sıfıra eşitliyoruz

    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Eğer bir konturun alanı 1000 pikselin altındaysa, o kontur göz ardı edilir
            continue

        (x, y, w, h) = cv2.boundingRect(contour)  # Konturun etrafında bir dikdörtgen çizer. x, y, w, h bu dikdörtgenin sol üst köşesinin koordinatlarını ve genişlik (w) ile yüksekliğini (h) temsil eder.
        center = (x + w // 2, y + h // 2)  # Dikdörtgenin merkez koordinatını hesaplar.

        # Bulunan kontur etrafına kırmızı bir dikdörtgen çizer.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Eğer bir önceki merkez (prev_center) varsa, nesnenin merkezindeki değişimi (dx, dy) hesaplarız.
        if prev_center is not None:
            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]

            # Speed (hız), bu değişimin büyüklüğüne ve fps değerine göre hesaplanır.
            speed = np.sqrt(dx**2 + dy**2) * fps

            # Angle (açı), nesnenin hareket ettiği yönü belirlemek için arctan2 fonksiyonu kullanılır.
            angle = np.degrees(np.arctan2(dy, dx))

            # Açıyı 0 ile 360 arasında tutmak için modül 360 ekleriz
            if angle < 0:
                angle += 360

        prev_center = center

    # Ekranda "Koordinatlar:", "Hiz:", "Aci:" ve "Ogrenci No:" yazıları gözükür ve yanlarında bu yazılara karşılık gelen değerler yazar
    cv2.putText(frame, f"Koordinatlar: ({center[0]}, {center[1]})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Hiz: {speed:.2f} px/s", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Aci: {angle:.2f} deg", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Ogrenci No: {ogrenci_numarasi}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Ekranda "Hareket Algılama Proje" adında bir pencere açılır
    cv2.imshow("Hareket Algilama Proje", frame)

    # "q" tuşuna basınca pencereyi kapatır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera bağlantısını ve diğer kaynakları serbest bırakır
cap.release()
# Herhangi bir pencere varsa kapatır
cv2.destroyAllWindows()
