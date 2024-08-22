import pandas as pd
import config
from preprocessor import Pipeline

pipeline = Pipeline(config.TARGET,
                    config.LEAKY_FEATURES,
                    config.DATA_TYPE_CONVERSION,
                    config.HIGH_LOW_CARDINALITY_FEATURES,
                    config.CAT_IMP_MODE,
                    config.CAT_IMP_MISSING,
                    config.CAT_VARS,
                    config.TEMPORAL_VARIABLES,
                    config.CONT_VARS)


if __name__ == "__main__":
    # load the dataset
    data = pd.read_csv(config.DATA_PATH)
    pipeline.fit(data)
    pipeline.evaluate()