# İmza və Mühür Analizi Tədqiqat Qeydi (Research Note)

## Mühüm Məhdudiyyətlər və Qeydlər
1. **Qələm Təzyiqi (Pen Pressure):** Skan olunmuş və ya rəqəmsallaşdırılmış kağız sənədlərdən fiziki qələm təzyiqini (pen pressure) etibarlı şəkildə ölçmək optiki olaraq qeyri-mümkündür. Bu səbəbdən sistem fiziki təzyiq əvəzinə piksellərin intensivlik və binarizasiya paylanmasından istifadə edir.
2. **Referans Tələbi (Reference Requirement):** Orijinal və ya etibarlı reference imza təmin edilmədən imzanın şəxsə məxsusluğu (autentikliyi) qəti şəkildə təsdiqlənə bilməz; nəticə `not_verified` olaraq qayıdır.
3. **Metodologiya:** SSIM, Hu Moments, Contour Matching və ORB xüsusiyyətləri ilə yanaşı rəqəmsal saxtakarlığı aşkar etmək üçün toxuma (texture) analizi tətbiq edilir.