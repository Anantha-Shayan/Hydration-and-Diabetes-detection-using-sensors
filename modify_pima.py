# Load PIMA dataset
columns = ["Glucose","BloodPressure","SkinThickness","Insulin"]
df = pd.read_csv('PIMA_diabetes.csv')
df.drop(columns, inplace=True, axis=1)
df.dropna(inplace = True)
df.tail()

# np.random.seed(42)

# --- Add HydrationLevel ---
# Default: High hydration for non-diabetics, low for diabetics
df["HydrationLevel"] = np.where(
    df["Outcome"] == 0,
    np.random.uniform(0.6, 1.0, len(df)),
    np.random.uniform(0.1, 0.5, len(df))
)

# --- Add exceptions ---
# 5% diabetics with high hydration
diabetic_ex = df[df["Outcome"] == 1].sample(frac=0.05, random_state=42).index
df.loc[diabetic_ex, "HydrationLevel"] = np.random.uniform(0.6, 0.9, len(diabetic_ex))

# 5% non-diabetics with low hydration
healthy_ex = df[df["Outcome"] == 0].sample(frac=0.05, random_state=99).index
df.loc[healthy_ex, "HydrationLevel"] = np.random.uniform(0.1, 0.5, len(healthy_ex))

# --- Add WaistCircumference ---
df["WaistCircumference"] = df["BMI"] * np.where(df["Outcome"] == 1, 2.5, 2.3)
df["WaistCircumference"] += np.random.normal(0, 2, len(df))  # small noise
df["WaistCircumference"] = df["WaistCircumference"].clip(60, 130).round(1)

# View a few rows
print(df.head())
