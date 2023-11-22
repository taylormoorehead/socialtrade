import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from gensim.models import KeyedVectors

word2vec_model = KeyedVectors.load_word2vec_format('word2vec_model.bin', binary=True)

healthcare_policies = ["policy 1"]
political_alignment_scores = [0.2]

def document_vector(word2vec_model, doc):
    words = [word for word in doc.split() if word in word2vec_model.vocab]
    if len(words) > 0:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

X = np.array([document_vector(word2vec_model, policy) for policy in healthcare_policies])
y = np.array(political_alignment_scores)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

mse = np.mean((model.predict(X_test) - y_test) ** 2)
print(f'Mean Squared Error: {mse}')

new_policy = "The Coronavirus Aid, Relief, and Economic Security Act, also known as the CARES Act, is a $2.2 trillion economic stimulus bill"
new_policy_vector = np.array([document_vector(word2vec_model, new_policy)])
predicted_alignment = model.predict(new_policy_vector)
print(f'Predicted Political Alignment: {predicted_alignment}')