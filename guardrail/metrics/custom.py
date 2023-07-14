class CustomMetricsWrapper:
    def __init__(self, custom_metric_function):
        self.custom_metric_function = custom_metric_function

    def evaluate(self, *args, **kwargs):
        return self.custom_metric_function(*args, **kwargs)
