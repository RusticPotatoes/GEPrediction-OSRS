import requests
import pandas as pd
#https://github.com/JonasHogman/osrs-prices-api-wrapper
class PricesAPI(object):
    def __init__(self, user_agent, contact):
        self.base_url = "https://prices.runescape.wiki/api/v1/osrs/"
        self.user_agent = {"User-Agent": user_agent, "From": contact}
        self._mappings = False

        self.times = ["5m", "1h", "3h", "6h", "24h"]

    def latest_df(self, mapping=False):
        prices = requests.get(self.base_url + "latest", headers=self.user_agent).json()["data"]
        prices = (pd.DataFrame.from_dict(prices)
                  .T
                  .fillna(0)
                  .astype("int")
                  .rename_axis("id")
                  )
        prices.index = prices.index.astype("int")
        if mapping:
            if not self._mappings:
                self._mappings = self.mapping_df()
            prices = self.__merge_mapping_df(prices)
        return prices

    def volumes_df(self, mapping=False):
        volumes = requests.get(self.base_url + "volumes", headers=self.user_agent).json()
        volumes = (
            pd.DataFrame(volumes["data"].items(), columns=["id", "volume"])
            .fillna(0)
            .astype("int")
            .set_index("id")
            .index.astype("int")
        )
        if mapping:
            if not self._mappings:
                self._mappings = self.mapping_df()
            volumes = self.__merge_mapping_df(volumes)
        return volumes


    def prices_df(self, time, mapping=False):
        if time in self.times:
            prices = requests.get(self.base_url + time, headers=self.user_agent).json()
            prices_data=prices["data"]
            prices_timestamp=prices['timestamp']
            prices_df = (pd.DataFrame.from_dict(prices_data)
                         .T
                         .fillna(0)
                         .astype("int")
                         .rename_axis("id")
                         .reset_index()
                         .rename(columns={"id":"item_id"})
                         .astype({"item_id": "int"})
                         )
            prices_df.index.name = 'index'
            prices_df['timestamp']=prices_timestamp
            #print(prices_df)
            if mapping:
                if not self._mappings:
                    self._mappings = self.mapping_df()
                prices_df = self.__merge_mapping_df(prices_df)
            return prices_df
        else:
            raise ValueError(f"Invalid timeframe selected, valid options are: {self.times}")


    def timeseries_df(self, step, id):
        if step in self.times:
            timeseries = requests.get(
                self.base_url + f"timeseries?timestep={step}&id={id}", headers=self.user_agent
            ).json()
            timeseries_data=timeseries["data"]
            item_id=timeseries['itemId']
            timeseries_data = (pd.DataFrame.from_dict(timeseries_data)
                          .fillna(0)
                          .astype("int")
                          .rename_axis("id")
                          )
            timeseries_data["item_id"]= item_id
            return timeseries_data
        else:
            raise ValueError(f"Invalid timeframe selected, valid options are: {self.times}")

    def mapping_df(self):
        mapping = requests.get(self.base_url + "mapping", headers=self.user_agent).json()
        #print(mapping)
        mapping = (
            pd.DataFrame.from_dict(mapping)
            .fillna(0)
            .rename(columns={"id":"item_id"})
            .set_index("item_id")
            .sort_index()
            .astype({"lowalch": "int", "limit": "int", "value": "int", "highalch": "int"}) 
            .reset_index()
            .astype({"item_id": "int"})
        )
        mapping.index.name = "index"
       # print(mapping)
        #mapping.index = mapping.index.map(int)
        #mapping = mapping.reset_index()
        #mapping = mapping.rename(columns={"id":"item_id"})
        #
        # mapping = mapping.rename(columns={"id":"item_id"})
        
                         #.sort_index()
                         #.reset_index()
                         #.rename(columns={"id":"item_id"})
        return mapping

    def __merge_mapping_df(self, data):
        data = data.merge(self._mappings, on="item_id")
        return data
