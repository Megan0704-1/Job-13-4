import os
from datetime import datetime
from .config import get_common_config


class MetricsLogger:
    def __init__(self):
        self.metrics_data = []
        self.log_path = get_common_config()["log_path"]
        self.date = str(datetime.now().date())

    def log_step(self, result):
        record = {
            "filename": os.path.basename(result.binary_path),
            "execution_time": result.execution_time,
            "binary_size": result.binary_size,
            "same_output": result.same_output,
        }

        self.metrics_data.append(record)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.metrics_data)

    def save_csv(self):
        df = self.to_dataframe()
        out_name = self.log_path + "/log-" + self.date
        df.to_csv(out_name, index=False)
