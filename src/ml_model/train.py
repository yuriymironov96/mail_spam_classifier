from sklearn.calibration import cross_val_predict
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

from fetch_data import load_dataset


def test_train_split(dataset):

    dataset['spam'] = dataset['Category'].apply(
        lambda x: 1 if x == 'spam' else 0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.Message,
        dataset.spam,
        test_size=0.2
    )

    return X_train, X_test, y_train, y_test


def no_preprocessing_categorical_bayes(X_train, X_test, y_train, y_test):

    v = CountVectorizer()

    train_x_cv = v.fit_transform(X_train.values)

    model = MultinomialNB()

    model.fit(train_x_cv, y_train)

    score = cross_val_score(
        model,
        train_x_cv,
        y_train,
        cv=3,
        scoring="accuracy"
    )
    print("cross_val_score", score)

    y_train_pred = cross_val_predict(model, train_x_cv, y_train, cv=3)

    f1 = f1_score(y_train, y_train_pred, average="macro")
    print("F1 score (macro):", f1)

    # cross_val_score [0.98250336 0.97779273 0.97037037]
    # F1 score (macro): 0.9507874558080206


dataset = load_dataset()
X_train, X_test, y_train, y_test = test_train_split(dataset)
no_preprocessing_categorical_bayes(X_train, X_test, y_train, y_test)
