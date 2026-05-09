"""ROBINS-I skeleton for non-randomized intervention studies."""

from __future__ import annotations

from datetime import datetime

from weighted_evidence.models import Identifier, Paper, RobinsIJudgment, StudyDesign
from weighted_evidence.rubric.robins_i import skeleton_robins_i


def _paper(abstract: str, design: StudyDesign = StudyDesign.cohort) -> Paper:
    return Paper(
        identifier=Identifier(pmid="x"),
        title="Comparative effectiveness study",
        abstract=abstract,
        publication_date=datetime(2023, 1, 1),
        design=design,
    )


def test_no_adjustment_yields_no_information_for_confounding() -> None:
    assess = skeleton_robins_i(_paper("A retrospective review of 200 patients."))
    confounding = next(d for d in assess.domains if d.name == "confounding")
    assert confounding.judgment == RobinsIJudgment.no_information


def test_adjustment_alone_is_serious() -> None:
    assess = skeleton_robins_i(
        _paper("Multivariable Cox regression adjusted for age, sex, and comorbidities.")
    )
    confounding = next(d for d in assess.domains if d.name == "confounding")
    assert confounding.judgment == RobinsIJudgment.serious


def test_adjustment_plus_sensitivity_reaches_moderate() -> None:
    assess = skeleton_robins_i(
        _paper(
            "Propensity score matching with E-value sensitivity analysis. "
            "Pre-specified primary outcome. Blinded outcome adjudication."
        )
    )
    confounding = next(d for d in assess.domains if d.name == "confounding")
    assert confounding.judgment == RobinsIJudgment.moderate


def test_overall_is_worst_domain() -> None:
    assess = skeleton_robins_i(_paper("A retrospective review of 50 patients."))
    judgments = [d.judgment for d in assess.domains]
    # No-information dominates when nothing is reported.
    assert RobinsIJudgment.no_information in judgments
