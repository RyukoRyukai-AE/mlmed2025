from src.data_preparation import Dataset
from src.model_build import Model
import joblib

data = Dataset('config.yaml')

train_path = data.get_path('raw/mitbih_train.csv')
test_path = data.get_path('raw/mitbih_test.csv')

train_df = data.read_data(train_path, 'Trainset')
test_df = data.read_data(test_path, 'Testset')

data.display(train_df)

y = train_df[187]
X = train_df.drop(columns=[187])

y_test = test_df[187]
X_test = test_df.drop(columns=[187])

model = Model (X, y)
model.balancing()

acc_train, acc_test, svc = model.build()

model.evaluate(X_test, y_test)

joblib.dump(svc, 'model/SVC_SMOTE.pkl')