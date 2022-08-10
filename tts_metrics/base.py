import abc
import json

class BasedMetric(metaclass=abc.ABCMeta):

    def __init__(self,
                 dataroot,
                 data_mapper_path,
                 metric_config,
                 name):
        """Base metric - interface for all metric class in project

        Args:
            dataroot (str): path to root dir of paths in data_mapper
            data_mapper_path (str): path to data_mapper (format is depend on metric)
            config (dataclass/custom class): provide hyperparams of 
            name (str): Name of metric
        """
        
        self.dataroot = dataroot
        self.data_mapper = json.load(open(data_mapper_path))
        self.metric_config = metric_config
        self.name = name
        
    def get_metric_name(self):
        """Return metric name
        """
        return self.name
    
    @abc.abstractmethod
    def compute(self):
        """Compute metric value
        """
        pass