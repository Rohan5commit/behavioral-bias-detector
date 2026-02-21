from sqlalchemy.orm import Session

from src.db.session import SessionLocal, engine
from src.models.database import Base, BiasScenario, LLMAgent
from src.scenarios.bias_templates import ScenarioGenerator
from src.utils.pit_controller import PointInTimeController


def seed_agents(db: Session) -> int:
    defaults = [
        # Default free-path setup: NVIDIA API key + OpenAI-compatible endpoint.
        {
            "model_name": "meta/llama-3.1-70b-instruct",
            "provider": "nvidia",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
    ]
    existing = {(row.provider, row.model_name) for row in db.query(LLMAgent).all()}
    inserted = 0
    for item in defaults:
        key = (item["provider"], item["model_name"])
        if key in existing:
            continue
        db.add(LLMAgent(**item))
        inserted += 1
    return inserted


def seed_scenarios(db: Session) -> int:
    generator = ScenarioGenerator(seed=42, pit_controller=PointInTimeController())
    generated = generator.generate_all_scenarios()
    existing_names = {row[0] for row in db.query(BiasScenario.scenario_name).all()}
    inserted = 0
    for data in generated:
        if data["scenario_name"] in existing_names:
            continue
        db.add(BiasScenario(**data))
        inserted += 1
    return inserted


def main() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        agents_inserted = seed_agents(db)
        scenarios_inserted = seed_scenarios(db)
        db.commit()
        print(f"Seed complete: agents_inserted={agents_inserted}, scenarios_inserted={scenarios_inserted}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
