#!/bin/bash
# Downloads chameleon_filtered.npz and squirrel_filtered.npz from Google Drive
# and places them in the expected data directories.

set -e

mkdir -p data/geom-gcn/chameleon
mkdir -p data/geom-gcn/squirrel

echo "Downloading geom-gcn folder from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf" -O data/tmp_geom_gcn

echo "Placing files..."
find data/tmp_geom_gcn -name "chameleon_filtered.npz" -exec cp {} data/geom-gcn/chameleon/ \;
find data/tmp_geom_gcn -name "squirrel_filtered.npz"  -exec cp {} data/geom-gcn/squirrel/ \;

rm -rf data/tmp_geom_gcn

echo "Done."
echo "chameleon: $(ls data/geom-gcn/chameleon/)"
echo "squirrel:  $(ls data/geom-gcn/squirrel/)"
