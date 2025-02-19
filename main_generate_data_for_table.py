import pandas as pd
import polars as pl
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from collections import Counter

def read_file(f):
    # Extracting the components from the filename
    collection, retrieval_model, dime, measure_name, nconf = f.rsplit("/", 1)[1].rsplit(".")[0].split("_")
    # Read the CSV using Polars (dtypes can be specified similarly to Pandas)
    df = pl.read_csv(f, schema_overrides={"query_id": pl.Utf8})
    # Add the new columns with the extracted values
    df = df.with_columns([
        pl.lit(collection).alias("collection"),
        pl.lit(retrieval_model).alias("retrieval_model"),
        pl.lit(dime).alias("dime"),
        pl.lit(nconf).alias("nconf")
    ])
    return df

if __name__ == "__main__":

    #src_path = '/hdd4/giuder/progetti/Eclipse/data/performance/all'
    src_path = '/hdd4/giuder/progetti/Eclipse/data/performance/active_feedback/all'
    #src_path = '/hdd4/giuder/progetti/Eclipse/data/performance/baselines/all'
    fn = list(glob(f"{src_path}/*.csv"))
    print(len(fn))

    #dimes_set = [['LLMEclipse'], ['PRFEclipse']]
    #dimes_set = [['GPTFilter'], ['TopkFilter']]
    dimes_set = [['ActiveFeedback'], ['NegActiveFeedback']]


    for dimes in dimes_set:
        # concat dataframe
        fn = list(filter(lambda x: np.any([d in x for d in dimes]), glob(f"{src_path}/*.csv")))
        perf = pl.concat([read_file(f) for f in tqdm(fn)], how="diagonal")
        
        dime = dimes[0]
        #filename = f'/hdd4/giuder/progetti/Eclipse/data/performance/tables/mega_files_{dime}_v2.parquet'
        filename = f'/hdd4/giuder/progetti/Eclipse/data/performance/tables/active_feedback_mega_files_{dime}_v2.parquet'
        #if filename not in os.listdir('/hdd4/giuder/progetti/Eclipse/data/performance/tables'): 
        print('saving file at: ', filename)
        perf.write_parquet(filename)

        # compute global average 
        perf = perf[['collection', 'retrieval_model', 'dime', 'nconf', 'query_id', 'alpha0', 'measure', 'value']]
        gloabal_average_perf = (
            perf.group_by(['collection', 'retrieval_model', 'dime', 'nconf', 'alpha0', 'measure'])
            .agg([
                pl.col("value").mean().alias("average_value")
            ])
            )   

        filename = f'/hdd4/giuder/progetti/Eclipse/data/performance/tables/active_feedback_global_performance_{dime}_v2.parquet'
        #filename = f'/hdd4/giuder/progetti/Eclipse/data/performance/tables/global_performance_{dime}_v2.parquet'
        #if filename not in os.listdir('/hdd4/giuder/progetti/Eclipse/data/performance/tables'): 
        print('saving file at: ', filename)
        gloabal_average_perf.write_parquet(filename)

        ## compute results
        #result = gloabal_average_perf.select(
        #    pl.all().top_k_by("average_value", k=5)
        #    .over(['collection', 'retrieval_model', 'dime', 'measure'], mapping_strategy="explode")
        #    )
        #
        #filename = f'/hdd4/giuder/progetti/Eclipse/data/performance/tables/rq1_top5_{dime}.parquet'
        ##if filename not in os.listdir('/hdd4/giuder/progetti/Eclipse/data/performance/tables'): 
        #print('saving file at: ', filename)
        #result.write_parquet(filename)