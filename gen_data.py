from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

metadata = SingleTableMetadata()
metadata.detect_from_csv('data/ss.csv')

datasets = load_csvs(
    folder_name='data/',
    read_csv_parameters={
        'skipinitialspace': True,
    })

synthesizer = GaussianCopulaSynthesizer(
    metadata,
)

synthesizer.fit(datasets['ss'])

synthetic_data = synthesizer.sample(
    num_rows=373
)

synthetic_data.to_csv('data/synthetic.csv', index=False)
