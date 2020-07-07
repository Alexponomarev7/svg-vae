import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
from tqdm import tqdm

glyph_ids = []
unis = []
widths = []
vwidths = []
splinesets = []
font_ids = []

with open('data_test', 'r') as fr:
    for line in tqdm(fr.readlines()):
        glyph_id, uni, width, vwidth, splineset, font_id = line.split('$')
        splineset = splineset.replace('\\n', '\n')

        if not ("EndSplineSet" in splineset):
            continue

        glyph_ids.append(int(glyph_id))
        unis.append(int(uni))
        widths.append(int(width))
        vwidths.append(int(vwidth))
        splinesets.append(splineset)
        font_ids.append(font_id)


df = pd.DataFrame({
    'uni': unis,
    'width': widths,
    'vwidth': vwidths,
    'sfd': splinesets,
    'id': glyph_ids,
    'binary_fp': font_ids
})

table = pa.Table.from_pandas(df)
pq.write_table(table, 'data_test.parquet')
