from pathlib import Path

import numpy as np
import pytest

from mggp import MGGP


@pytest.mark.parametrize("metric", ["accuracy", "log_loss"])
def test_classifier_generalizes_and_checkpoint_reproduces_held_out_scores(classification_data, metric):
    x, xv, y, yv = classification_data
    engine = MGGP(inputs=x, outputs=y, validation=(xv, yv), mode="FIR",
                  problem_type="classification", classification_metric=metric,
                  evaluationType="INSTANT", evaluationTypeTest="INSTANT",
                  generations=2, populationSize=20, nTerms=10, maxHeight=2,
                  nDelays=1, mutationRate=.2, crossoverRate=.9, filename="classifier")
    best, = engine.run()
    assert Path("models_saved/classifier_0.pkl").is_file()
    prediction, target = best.predict_proba("INSTANT", yv, xv)
    reloaded, _ = engine.load_model().predict_proba("INSTANT", yv, xv)
    np.testing.assert_allclose(prediction, reloaded)
    true_labels = target.argmax(axis=1)
    if metric == "accuracy":
        score = np.mean(true_labels == prediction.argmax(axis=1))
        majority = np.max(np.bincount(true_labels)) / len(true_labels)
        assert score > majority
    else:
        likelihood = prediction[np.arange(len(target)), true_labels]
        training_priors = y.mean(axis=0)
        assert -np.log(likelihood).mean() < -np.log(training_priors[true_labels]).mean()
