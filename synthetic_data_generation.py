import numpy as np
import pandas as pd

"""--------------------------------SYNTHETIC DATA GENERATION--------------------------------"""

n = 10000  # number of samples

age = np.random.randint(20, 80, n)  #random 1000 int between 20,80
bmi = np.random.normal(27, 5, n)  #mean=27, std=5
pregnancies = np.random.poisson(2, n) #avg 2 pregnancies
#thickness = np.random.uniform(10, 50, n)  #values with equal chances between 10,50
hydration = np.clip(np.random.normal(0.6, 0.15, n), 0.1, 1.0) #mean=0.6, std=0.15
# clipped to 0.1 , 1.0 (i.e, values <0.1 and >1.0 is set to 0.1 and 1.0 respectively)

#Assign labels(0/1) for diabetes using probability
prob = (
    0.15 +  #This (0.15) is known as the base probability or intercept,similar to the bias term in a machine learning model.
            #It means even a healthy person (normal BMI, hydrated, young, etc.) has a small chance of diabetes.
    0.25*(bmi > 30) +
    0.25*(hydration < 0.4) +
    0.20*(age > 50) +
    #0.10*(thickness > 30) +
    0.15*(pregnancies > 3)
    # pregnancies and thickness moderately affect diabetes prediction (so 10%)
    )
diabetes = np.random.binomial(1, np.clip(prob, 0, 1))

#save data
df = pd.DataFrame({
    'age': age,
    'bmi': bmi,
    'pregnancies': pregnancies,
    #'thickness': thickness,
    'hydration': hydration,
    'diabetes': diabetes
})
df.to_csv("synthetic_diabetes_data.csv", index=False)
