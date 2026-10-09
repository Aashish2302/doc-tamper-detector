#!/bin/bash
# query the running server
curl -s -X POST "http://localhost:8080/predict?threshold=0.5" -F "file=@$1" | python3 -c "import sys,json;d=json.load(sys.stdin);print('tampered:',d['tampered'],'| max_score:',round(d['max_score'],3),'| tampered_px:',d['tampered_pixels'])"
