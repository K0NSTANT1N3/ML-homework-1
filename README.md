# ML-homework-1
Repository for ML homework 1

---

## პროექტის სათაური

Kaggle-ის კონკურსის მოკლე მიმოხილვა

მოცემულ დავალებაში მოცემული გვაქვს სახლების ინფრომაცია რაზე დაყრდნობითაც უნდა 
ვივარაუდოთ რა ფასი ექნება თითოეულ სახლს.

---

## თქვენი მიდგომა პრობლემის გადასაჭრელად

პირველ რიგში დავიწყე მონაცემების კვლევით და ვიზუალიზაციით, რათა უკეთ გავრკვეულიყავი მონაცემებში და მათ შორის არსებულ ურთიერთობებში.  
კარგად წავიკითხე `data_description.txt` ფაილი სადაც აღწერილია თითოეული სვეტის მნიშვნელობა.

ამაზე დაყრდნობით ვაპირებდი მონაცემების დამუშავებას:  
- ზოგი სვეტის წაშლა  
- ზოგი ცვლადის გაერთიანება ერთ ცვლადში უფრო ინფორმაციული სვეტების მისაღებად  
- კატეგორიულების რიცხვითში გადაყვანა  
- ზოგი რიცხვითი პირიქით კატეგორიულში გადაყვანა  

---

## რეპოზიტორიის სტრუქტურა

- ყველა ფაილის განმარტება  
- `/dumb-model` – საბაზისო წრფივი მოდელის გაშვება დაუმუშავებელ მონაცემებზე  
- `/model-experiment` – მთავარი კოდი, ფოლდერში:  
  - `1_eda.ipynb` – ვიზუალიზაცია  
    1. სახლების ფასების განაწილება  
    2. სვეტები სადაც მონაცემების ძალიან მწირი ნაწილია შევსებული  
    3. სვეტების კორელაცია საბოლოო ფასთან  
    4. სვეტების კორელაცია ერთმანეთთან  
    5. ნულთან ახლო ვარიაციის მქონე სვეტები  
  - `2_cleaning_features.ipynb` – მონაცემების გასუფთავება, ვალიდაციის და ტრენინგის ნაწილი  
  - `3_feature_selection.ipynb` – საბოლოო feature-ების არჩევა  

---

## Feature Engineering

Feature engineering-ში შევეცადე მაქსიმალურად informativ და clean მონაცემების შექმნას. ყველაფერი რიგად გავაკეთე:

### Drop useless columns
- High missing columns (>80% missing) გადავყარე, რადგან ვერ მოგვცემდნენ რაიმე ინფორმაციას  
- Near zero variance სვეტები (>95% same value) ასევე არ არის informativ, ამიტომაც წავაშლე

### Fix wrong data types
- MSSubClass, MoSold, YrSold – numbers უბრალოდ label იყო, გადავაქციე string-ად (categorical)  
- ეს რომ არ მექნა მოდელს ეგონებოდა, რომ მაღალი MSSubClass-ი უფრო ძვირია, რაც არ არის სწორი

### HousePreprocessor Pipeline

Pipeline-ში გარდავქმნი ყველა კატეგორიულს რიცხვითში და ვიღებ საბოლოო dataset-ს ტრენინგისთვის.

**რას ვაკეთებ პაიპლაინში:**

1. **Define nominal columns** – ყველა categorical სვეტი, რომელიც მოგვარდება one-hot encoding-ით  
2. **Learn medians for numeric columns** – ყველა რიცხვითი სვეტისთვის მედიანით ვავსებ ნალებს  
3. **Learn modes for remaining categorical columns** – დანარჩენი categorical სვეტები ვავსებ მოდით  
4. **Learn OHE categories** – train-ის data-ზე მხოლოდ, რათა მომავალში unseen categories პრობლემა არ შექმნან  

**Handle missing values smartly**  
- Fill “None”: სვეტები სადაც NA ნიშნავს არარსებობას (Fireplace, Garage, Basement qualities)  
- Fill “0”: სვეტები სადაც NA ნიშნავს 0 area, 0 cars, 0 baths  
- LotFrontage: NA filled by neighborhood median, fallback: გლობალური მედიანა  
- Remaining numeric NAs: სვეტის მედიანით შევსება  
- Remaining categorical NAs: ვავსებ მოდით  

**Ordinal encoding**  
- Quality columns (ExterQual, BsmtQual...) mapped Po < Fa < TA < Gd < Ex  
- BsmtExposure, BsmtFinType1/2, GarageFinish, Functional, LotShape, LandSlope, PavedDrive – რიცხვითში გადამყავს ზრდადობით  

**Create new features**  
- `TotalSF` = TotalBsmtSF + 1stFlrSF + 2ndFlrSF  
- `TotalBaths` = FullBath + BsmtFullBath + 0.5 * (HalfBath + BsmtHalfBath)  
- `TotalPorchSF` = ჯამი: OpenPorch, EnclosedPorch, 3SsnPorch, ScreenPorch  
- `HouseAge` = YrSold - YearBuilt  
- `RemodAge` = YrSold - YearRemodAdd  
- `IsRemodeled` = 1 თუ გარემონტებულია, 0 თუ არა  
- `HasGarage`, `HasBasement`, `HasFireplace`, `HasPool` = 1 თუ ფართობი >0, 0 თUარადა  

**One-hot encoding**  
- Nominal columns converted to OHE  
- ყველა ნომინალური სვეტი დავდრობე, ახალი ორობითი სვეტები შევქმენი თითოეული კატეგორიისთვის  

**Final safety check**  
- თუ რამე string სვეტი დამრჩა, გადავყარე  

**შედეგად:**  
მივიღე გასუფთავებული, რიცხვითი ბაზა, სადაც არ არის missing values, კატეგორიულები არ მაქვს, ახალი features დავამატე და ყველა სვეტი მზად არის მოდელირებისთვის  

---

## Feature Selection

Feature selection-ში ვცდილობ საუკეთესო, informativ სვეტების არჩევას.

### კორელირებული სვეტების ამოშლა
- თუ ორი სვეტი ერთმანეთთან ძალიან მაღალი კორელაცია (>0.85) აქვს, ერთ-ერთი ზედმეტია  
- შევარჩიე ის, რომელიც target-თან უფრო მაღალი კორელაციაშია, დანარჩენი წავშალე  
- ამ გზით შევამცირე redundancy და overfitting-ის რისკი  

### ნულთან ახლო ვარიაციის მქონე სვეტების ამოშლა
- სვეტები, სადაც თითქმის ყველა სახლი ერთი და იგივე მნიშვნელობას იღებს, არ არის informativ  
- VarianceThreshold-ის გამოყენებით მოვაშორე სვეტები სადაც ვარიაცია < 0.01  
- შედეგად, მხოლოდ განსხვავებულ სვეტებს ვტოვებ, რომლებიც რეალურად ფასის განსაზღვრაში მონაწილეობენ  

### SelectKBest
- თითოეული სვეტის სტატისტიკური მნიშვნელობა target-თან შევაფასე `f_regression`-ის საშუალებით  
- შევარჩიე top 100 მახასიათებელი, რომელიც ყველაზე მეტ ინფორმაციას მაძლევს  
- ამ გზით ვიღებ უფრო ეფექტურ სვეტებს  

### RFE (Recursive Feature Elimination)
- ეს ყველაზე წარმატებული მიდგომა აღმოჩნდა ჩემს შემთხვევაში  
- RandomForestRegressor-ით ვმუშაობდი estimator-ის როლში  
- RFE თითო iteration-ში რამდენიმე ნაკლებ informativ feature-ს ამოშლიდა და ბოლოს რჩებოდა 100 საუკეთესო სვეტი  
- ამ მეთოდით მივიღე ყველაზე კარგი შედეგი, ამიტომ საბოლოო მოდელში RFE-ს ვიყენებ  

### ყველა ვერსიის შენახვა
- ყველა ნაბიჯის შემდეგ სხვადასხვა ვერსია შევინახე:  
  - ყველა სვეტი (baseline)  
  - კორელაციით დაქორექტირებული  
  - variance-ის მიხედვით დაქორექტირებული  
  - SelectKBest-ის მიხედვით  
  - RFE-ის მიხედვით (**best result**)  

### MIX
- ყველა ზემოთხსენებული მეთოდი მოვსინჯე, რომ ნაკლები დრო დამჭირვებოდა RFE-ზე  
- რაც რეალურად გადასაგდები იყო, ის უკვე გადამეგდო  
- წმინდა RFE-ს მაინც ჩამორჩება, დადებითი ის აქვს რომ დროს ცოტათი იგებ  

**შედეგად:**  
RFE-ის გამოყენებით მივიღე საუკეთესო მახასიათებლები, რომლებზეც ვასწავლი მოდელს  
ყველა ვერსია შენახულია `../data` ფოლდერში, რათა შემდგომში ექსპერიმენტებისთვის გამოვიყენო  

---

## Training

- ტესტირებული მოდელები  
- Hyperparameter ოპტიმიზაციის მიდგომა  
- საბოლოო მოდელის შერჩევის დასაბუთება  

---

## MLflow Tracking

- MLflow ექსპერიმენტების ბმული  
- ჩაწერილი მეტრიკების აღწერა  
- საუკეთესო მოდელის შედეგები