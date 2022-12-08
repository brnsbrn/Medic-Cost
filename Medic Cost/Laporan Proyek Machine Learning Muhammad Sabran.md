# Laporan Proyek Machine Learning Muhammad Sabran

## Domain Proyek

Banyak orang takut untuk pergi berobat ke rumah sakit. Entah dengan alasan sakit yang masih belum parah atau biaya yang mahal untuk berobat. Seperti dilansir oleh CNBC [Mahalnya Biaya Berobat, Dari Jatuh Sakit Bisa Jatuh Bangkrut](https://www.cnbcindonesia.com/lifestyle/20180429164515-33-12936/mahalnya-biaya-berobat-dari-jatuh-sakit-bisa-jatuh-bangkrut). Bahkan hal pertama yang mereka pikirkan ketika sakit bukan tentang sembuh atau tidaknya, tapi mengenai biaya yang harus dikeluarkan. Untuk itu diperlukan penelitian untuk memprediksi biaya berobat di rumah sakit seperti yang dilakukan oleh **Mohamad Arif Muharam** [Prediksi Biaya Medis](https://www.bisa.ai/portofolio/detail/Mzgw). Proyek kali ini bertujuan untuk memprediksi nilai medical cost dari yang dikeluarkan seseorang untuk berobat ke rumah sakit.  
## Business Understanding

Biasanya ketika orang sakit, selama sakitnya belum parah, ia akan terus mengabaikan dan meremehkan sakitnya. Namun, ketika sakitnya sudah bertambah parah dan sangat berbahaya, barulah ia mau untuk pergi berobat ke rumah sakit. Sayangnya, ketika sakitnya sudah parah dan kronis tentunya membutuhkan biaya yang besar untuk mengobatinya. Andaikan dari awal ketika sakitnya masih belum parah dia sudah pergi berobat, pastinya biaya yang dikeluarkan tidak terlalu besar. Untuk itu dengan adanya sistem prediksi biaya perobatan ini dapat mengurangi kekhawatiran pasien untuk pergi berobat ke rumah sakit.
### Problem Statement:
Berdasarkan permasalahan di atas, maka permasalahan yang ditemukan yaitu.
- Apakah faktor yang memengaruhi besarnya biaya pengobatan?
- Bagaimana cara memprediksi biaya pengobatan?

### Goals:
Maka berdasarkan permasalahan di atas, adapun tujuan dari proyek ini yaitu.
-   Mengetahui faktor yang mempengaruhi biaya pengobatan.
-	Memprediksi besarnya biaya pengobatan.

### Solution Statements:
-	Melakukan proses EDA untuk mengetahui korelasi dari setiap fiturnya.
-	Membuat Model Machine Learning dengan menggunakan beberapa algoritma antara lain:
	1.	KNN
	2.	Random Forest

## Data Understanding

Adapun dataset yang digunakan diperoleh melalui situs kaggle yang dapat diunduh melalui [tautan](https://www.kaggle.com/datasets/mirichoi0218/insurance). Dataset ini memiliki 1339 data dan 7 fitur. Fitur yang terdiri dari fitur numerikal (age, bmi, children, dan charges) serta fitur kategorikal (sex, smoker, dan region).
Adapun penjelasan detail mengenai fitur-fitur yang ada di dataset tersebut yaitu.
- age : Umur pasien
- sex : Jenis kelamin pasien
- bmi : Body Mass Index (berat badan normal/sehat)
- children : Jumlah anak yang ditanggung oleh asuransi
- smoker : Merokok atau tidak
- region : Tempat asal (northeast, southeast, southwest, northwest)
- charges : Biaya medis individu (per orang) 

### EDA Univariate


![download](https://user-images.githubusercontent.com/113587270/190840585-0a68e9d9-03ff-4ca8-80ab-dcfb41e45e02.png)


Berdasarkan grafik di atas jumlah laki-laki dan perempuan tidak terpaut jauh, meskipun didominasi oleh laki-laki.



![image](https://user-images.githubusercontent.com/113587270/190840797-3693aae9-892f-45aa-b01c-e57af08d932d.png)


Berdasarkan grafik di atas, sebesar 79.5% pasien bukan merupakan perokok.


![image](https://user-images.githubusercontent.com/113587270/190840840-51cf5bfb-c5dc-49e4-b238-b73b39407cf3.png)

Berdasarkan grafik di atas, mayoritas pasien berasal dari southeast.



### EDA Multivariate

![image](https://user-images.githubusercontent.com/113587270/190840953-4d3fee47-0380-4065-9757-9eb94220f11b.png)

![image](https://user-images.githubusercontent.com/113587270/190840960-217d6ef1-6d61-4a53-b6fd-e5fd06ace887.png)

![image](https://user-images.githubusercontent.com/113587270/190840910-7ca87982-5635-4afb-af2b-c153dd64dd49.png)

Berdasarkan ketiga grafik di atas, dapat disimpulkan bahwa yang sangat menentukan besarnya charges ialah status merokok atau tidak. Di mana apabila status pasien adalah seorang perokok, maka biaya pengobatannya akan lebih besar diandingkan dengan pasien yang tidak merokok.

**Korelasi Matriks**


![image](https://user-images.githubusercontent.com/113587270/190841074-9dc3e922-ab7a-4834-a2dc-d054c3ebb5f8.png)

Berdasarkan matriks di atas, fitur numerikal memiliki korelasi yang rendah terhadap charges. Maka kemungkinan fitur yang berkorelasi kuat ada di fitur kategorikal.


## Data Preparation

### Melakukan Encoding
Langkah pertama yaitu melakukan one-hot-encoding pada fitur kategorikal (sex, smoker, dan region) menggunakan get_dummies.

![image](https://user-images.githubusercontent.com/113587270/190841224-6de0a419-812a-4513-b3ba-980f8190450c.png)

Selanjutnya buat kembali korelasi matriks seluruh fitur yang ada.

![image](https://user-images.githubusercontent.com/113587270/190841288-1606ef3c-2db4-43cd-8855-39ae0c062203.png)

Maka didapatkan fitur yang berkorelasi kuat terhadap charges ialah fitur smoker (yes dan no).

### Melabeli Data
Yang pertama membuat dataframe X yang menampung variabel independen, caranya cukup dengan drop variabel dependen (charges).

![image](https://user-images.githubusercontent.com/113587270/190841382-ab2d2c03-58ce-46fb-b5c1-14619baa1539.png)

Selanjutnya buat dataframe y untuk menampung variabel dependen (charges)

![image](https://user-images.githubusercontent.com/113587270/190841406-677d74e1-b613-456a-bf9f-1a7105aaf715.png)

### Train-Test-Split
Selanjutnya membagi data sampel menjadi data train dan data test, dengan porsi 85% data train dan 15% data tes.

![image](https://user-images.githubusercontent.com/113587270/190841552-0196cf84-804c-4f00-bcbf-aa8c680868c2.png)

![image](https://user-images.githubusercontent.com/113587270/190841575-c2343b26-f8ac-4098-8ffc-f9ad4bef0ece.png)

### Standarisasi
Selanjutnya melakukan standarisasi menggunakan StandardScaler dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. Sehingga nilai standar deviasi sama dengan 1 dan mean sama dengan 0. 

![image](https://user-images.githubusercontent.com/113587270/190841698-6f137525-a6a4-40a1-8b50-124d100effe7.png)

![image](https://user-images.githubusercontent.com/113587270/190841834-55150fa6-9631-4d1c-a9df-94fdd16f5e4e.png)





## Modeling
Pada tahap ini, kita akan mengembangkan model machine learning dengan 2 algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan, antara lain:

-	K-Nearest Neighbor
-	Random Forest

Sebelum itu buat dataframe untuk menganalisis model.

![image](https://user-images.githubusercontent.com/113587270/190846280-3057b37e-db08-42e2-9db9-a6dcea2d619b.png)


1.	_**KNN**_
    merupakan algoritma yang berfungsi untuk melakukan klasifikasi suatu data berdasarkan data pembelajaran (train data sets), yang diambil dari k tetangga terdekatnya (nearest neighbors). Kelebihan dari algoritma ini yaitu mudah diimplementasi untuk menjadi sebuah model, namun kekurangannya model ini kurang efektif untuk data dalam jumlah besar.
    
    ![image](https://user-images.githubusercontent.com/113587270/190842113-e5203a0b-6dd5-4516-a112-eeb16274eee5.png)
   
    Disini saya menggunakan parameter jumlah tetangga sebanyak 5, artinya ia akan mengambil 5 tetangga dengan jarak terdekat (Menggunakan Euclidean Distance). Selanjutnya dia akan mengambil data mayoritas yang ada di 5 sampel tetangga tersebut untuk dimasukkan menjadi data baru. Pada tahap ini kita hanya melatih data training dan menyimpan data testing.

    
2.	_**Random Forest**_
   Adalah algoritma dalam _machine learning_ yang digunakan untuk pengklasifikasian dataset dalam jumlah besar. Karena fungsinya bisa digunakan untuk banyak dimensi dengan berbagai skala dan performa yang tinggi. Klasifikasi ini dilakukan melalui penggabungan tree dalam decision tree dengan cara training dataset yang Anda miliki. Nantinya ia akan menggabungkan beberapa decision tree. Nantinya random forest akan mencari fitur terbaik secara acak, fitur terbaik inilah yang akan berperan penting dalam hasil prediksi modelnya. Kelebihan algoritma ini yaitu mampu mengatasi data non linear, sedangkan kekurangannya yaitu membutuhkan waktu yang lama pada saat di training.
   
    ![image](https://user-images.githubusercontent.com/113587270/190842430-45c2cffc-f859-46e8-a5f9-61491c16de88.png)
   
    Berikut parameter yang saya gunakan.
-	n_estimator: jumlah trees (pohon) di forest. Di sini saya set n_estimator=100.
-	max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. Di sini saya set max_depth=16.
-	random_state: digunakan untuk mengontrol random number generator yang digunakan. 
-	n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.


## Evaluasi

Matriks evaluasi yang saya gunakan ialah MSE (Mean Squared Error) dan R2 Square

_**MSE**_
Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut.

![image](https://user-images.githubusercontent.com/113587270/190846428-72368a8a-68f0-41db-bc24-b703e9904c40.png)

N = jumlah dataset
yi = nilai sebenarnya
y_pred = nilai prediksi

Sebelum menghitung MSE, lakukan scaling pada fitur numerik di data test.

![image](https://user-images.githubusercontent.com/113587270/190846552-2bae65c5-2f2b-4e14-8219-5727145982df.png)

Lakukan evaluasi kedua model dengan menggunakan matriks MSE. Saat menghitung nilai Mean Squared Error pada data train dan test, kita membaginya dengan nilai 1e3. Hal ini bertujuan agar nilai mse berada dalam skala yang tidak terlalu besar.

![image](https://user-images.githubusercontent.com/113587270/190846704-f0f316d9-50cb-433f-b33c-3bdf3b97d188.png)

Hasil evaluasi pada data latih dan data test menggunakan matriks MSE adalah sebagai berikut.

![image](https://user-images.githubusercontent.com/113587270/190842958-d600e915-dc97-462b-b293-b85f1e66f779.png)

![image](https://user-images.githubusercontent.com/113587270/190843458-664ae97f-da2c-4fcc-aa4c-e5f99baa6f38.png)




_**R2 Square**_
Nilai R-squared (R2) digunakan untuk menilai seberapa besar pengaruh variabel laten independen tertentu terhadap variabel laten dependen. Terdapat tiga kategori pengelompokan pada nilai R square yaitu kategori kuat, kategori moderat, dan kategori lemah [2].

![image](https://user-images.githubusercontent.com/113587270/190843383-96d94ca4-a0ea-437b-ade5-af2f9396264d.png)

Lakukan evaluasi kedua model menggunakan matriks R2 Square

![image](https://user-images.githubusercontent.com/113587270/190846792-0f673b1a-f240-408a-a8cb-f532e101c3ea.png)

Hasil evaluasi pada data latih dan data test menggunakan matriks R2 Square adalah sebagai berikut.

![image](https://user-images.githubusercontent.com/113587270/190843508-5a7a8a10-0f23-4a0b-a703-883caade8334.png)



Berdasarkan hasil evaluasi matriks MSE, nilai error dari KNN jauh lebih besar dibandingkan dengan Random Forest. Sedangkan berdasarkan hasil evaluasi matriks R2 Square didapatkan bahwa nilai R2 Square dari Random Forest lebih besar daripada KNN.

Kita uji dengan melakukan prediksi.

![image](https://user-images.githubusercontent.com/113587270/190843580-66a1bfcd-fb92-4cae-8f53-41d49f8f4c5a.png)

Dapat dilihat bahwa prediksi dari Random Forest kebanyakan lebih mendekati dengan nilai y. Sehingga dapat disimpulkan Random Forest lebih efektif digunakan dalam kasus ini.



## Kesimpulan

-	Dapat membuat prediksi mengenai biaya pengobatan rumah sakit dengan prediksi sebagai berikut dengan lebih efektif menggunakan algoritma Random Forest.

	![image](https://user-images.githubusercontent.com/113587270/190843580-66a1bfcd-fb92-4cae-8f53-41d49f8f4c5a.png)


-	Berdasarkan korelasi matriks, fitur smoker yang sangat mempengaruhi besar atau tidaknya charges.

## Referensi

[1]	Hair, Jr., Joseph F., et. al. (2011). Multivariate Data Analysis. Fifth Edition. New Jersey: PrenticeHall, Inc.
