import json


class ModelSelector:
    def __init__(self, model_name, data_feature, dataset_name):
        self.model_name = model_name
        self.selected_model = None
        self.data_feature = data_feature
        self.dataset_name = dataset_name
        self.select_model()

    def select_model(self):
        print("Selecting model: " + self.model_name)
        if self.model_name == 'ReDyNet':
            from model.MyModel.ReDyNet import ReDyNet
            config = {}
            config_path = 'model/MyModel/' + self.dataset_name + '.json'
            print("-------------------config-------------------")
            with open(config_path, 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in config:
                        config[key] = x[key]
                        print(key + " : " + str(x[key]))
            print("--------------------------------------------")

            self.selected_model = ReDyNet(config=config, data_feature=self.data_feature)

        else:
            raise ValueError('Model not found.')

    def get_model(self):
        return self.selected_model
