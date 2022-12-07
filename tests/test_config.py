from nlg_analysis.cfg import AnalysisConfig, TrainConfig


class TestConfig:
    def test_analysis_config(self):
        analysis_config = AnalysisConfig()
        assert analysis_config.spacy_model == "pl_core_news_lg"

    def test_train_config(self):
        train_config = TrainConfig()
        assert train_config.gpt_model
        assert train_config.gpt_model_name == "flax-community/papuGaPT2"
        assert train_config.block_size == 64

    def test_config_from_dict(self):
        train_config = TrainConfig()
        assert train_config.epochs == 25
        test_dict = {"epochs": 1000}
        test_config = TrainConfig.from_dict(test_dict)
        assert test_config.epochs == 1000
