import os
from mggp import MGGP
import numpy as np

def test_classification_runs(classification_data):

    X_train, X_test, y_train, y_test = classification_data

    model_file = "test_classification.pkl"

    mggp = MGGP(
        inputs=X_train,
        outputs=y_train,
        validation=(X_test, y_test),
        problem_type="classification",
        classification_metric="accuracy",
        evaluationType='INSTANT',
        evaluationTypeTest='INSTANT',
        nDelays=1,
        k=1,
        generations=2,
        populationSize=20,
        nTerms=10,
        maxHeight=2,
        mutationRate=0.2,
        crossoverRate=0.9,
        filename=model_file,
        mode="FIR",
    )

    mggp.run()

    assert os.path.exists(model_file)

def test_classification_logloss_runs(classification_data):

    X_train, X_test, y_train, y_test = classification_data

    model_file = "test_classification.pkl"

    mggp = MGGP(
        inputs=X_train,
        outputs=y_train,
        validation=(X_test, y_test),
        problem_type="classification",
        classification_metric='log_loss',
        evaluationType='INSTANT',
        evaluationTypeTest='INSTANT',
        nDelays=1,
        k=1,
        generations=2,
        populationSize=20,
        nTerms=10,
        maxHeight=2,
        mutationRate=0.2,
        crossoverRate=0.9,
        filename=model_file,
        mode="FIR",
    )

    mggp.run()

    assert os.path.exists(model_file)


# def test_classification_predict_shape(classification_data):
#
#     X_train, X_test, y_train, y_test = classification_data
#
#     mggp = MGGP(
#         inputs=X_train,
#         outputs=y_train,
#         validation=(X_test, y_test),
#         problem_type="classification",
#         classification_metric="accuracy",
#             evaluationType='INSTANT',
#             evaluationTypeTest='INSTANT',
#         generations=2,
#         populationSize=20,
#     )
#
#     mggp.run()
#
#     y_pred = mggp.predict(X_test)
#
#     assert y_pred.shape[0] == X_test.shape[0]
#
# def test_classification_probability_sum(classification_data):
#
#     X_train, X_test, y_train, y_test = classification_data
#
#     mggp = MGGP(
#         inputs=X_train,
#         outputs=y_train,
#         validation=(X_test, y_test),
#         problem_type="classification",
#         classification_metric="accuracy",
#             evaluationType='INSTANT',
#             evaluationTypeTest='INSTANT',
#         generations=2,
#         populationSize=20,
#     )
#
#     mggp.run()
#
#     y_pred = mggp.predict(X_test)
#
#     np.testing.assert_allclose(
#         y_pred.sum(axis=1),
#         np.ones(len(y_pred)),
#         atol=1e-6,
#     )
#
#
# def test_classification_better_than_random(classification_data):
#
#     X_train, X_test, y_train, y_test = classification_data
#
#     mggp = MGGP(
#         inputs=X_train,
#         outputs=y_train,
#         validation=(X_test, y_test),
#         problem_type="classification",
#         classification_metric="accuracy",
#             evaluationType='INSTANT',
#             evaluationTypeTest='INSTANT',
#         generations=5,
#         populationSize=50,
#     )
#
#     mggp.run()
#
#     acc = mggp._hof[0].Fitness
#
#     assert acc > 0.50