from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset

metadata = SingleTableMetadata()
metadata.detect_from_csv('data/ss.csv')
datasets = load_csvs(
    folder_name='data/',
    read_csv_parameters={
        'skipinitialspace': True,
    })

synthesizer = SingleTablePreset(
    metadata,
    name='FAST_ML'
)

synthesizer.fit(datasets['ss'])

synthetic_data = synthesizer.sample(
    num_rows=1000
)

synthetic_data.to_csv('data/synthetic.csv', index=False)
