import pandas as pd
from sklearn.model_selection import train_test_split
import os
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('StudentData.csv')

# Select numerical columns
numerical_df = df.select_dtypes(include=['float64','int64'])

# Calculate correlation matrix from numerical columns
correlation_matrix = numerical_df.corr()

# Identify columns
high_corr_columns = [col for col in correlation_matrix.columns if any(abs(correlation_matrix[col]) > 0.8) and col != 'Grades']

print(f"Dropped comlumns: {high_corr_columns}\n")

df = df.drop(columns=high_corr_columns)

# Remove spaces and replace with _
df['Parental_Education'] = df['Parental_Education'].replace({
    'High School': 'High_School',
    'Some College': 'Some_College'
})

# Categorize Family Income
df['Family_Income'] = pd.to_numeric(df['Family_Income'], errors='coerce')
income_bins = [30000.0, 39000.0, 48000.0, 57000.0, 66000.0, 75000.0]
income_labels = ["LOW", "LOWER_MIDDLE", "MIDDLE", "UPPER_MIDDLE", "HIGH"]
# Converting numerical data to string categories
df['Family_Income_Category'] = pd.cut(df['Family_Income'], bins=income_bins, labels=income_labels, include_lowest=True)

# Remove numerical columns
df = df.drop(columns=['Family_Income'])

# Delete rows with too many missing values
df = df.dropna(thresh=7)

# Fill missing with most frequent value in column
for col in df.select_dtypes(include=['object', 'category']):
    df[col] = df[col].fillna(df[col].mode()[0])
    
# Convert columns to string
df = df.astype(str)

# Set target values
X = df.drop(columns=['Grades'])
y = df['Grades']

# Split the data (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save train and test data
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
# Pathing
prepData_path = '../PrepData'
train_file = os.path.join(prepData_path, 'Stdnt_Train.csv')
test_file = os.path.join(prepData_path, 'Stdnt_Test.csv')
train_oversampled_file = os.path.join(prepData_path, 'Stdnt_Oversampled_Train.txt')
test_oversampled_file = os.path.join(prepData_path, 'Stdnt_Oversampled_Test.txt')

# Create PrepData if needed
ensure_directory_exists(prepData_path)

# Save Train/Test csv
X_train.assign(Grades=y_train).to_csv(train_file, index=False)
X_test.assign(Grades=y_test).to_csv(test_file, index=False)

# Oversampling function
def oversample_data(X, y, target_size, filename):
    current_size = len(y)
    multiplier = target_size // current_size
    oversampler = RandomOverSampler(sampling_strategy={label: multiplier * count for label, count in y.value_counts().items()})
    X_oversampled, y_oversampled = oversampler.fit_resample(X, y)
    oversampled_data = X_oversampled.assign(Grades = y_oversampled)
    oversampled_data.to_csv(filename, sep=' ', header=False, index=False)
    print(f"Oversampled dataset size: {len(oversampled_data)} rows")
    
# Apply oversampling for train and test sets
oversample_data(X_train, y_train, 500000, train_oversampled_file)
oversample_data(X_test, y_test, 500000, test_oversampled_file)

print(f"Arquivos salvos em: {prepData_path}")