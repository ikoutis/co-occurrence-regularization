import urllib.request
import re
html = urllib.request.urlopen("https://data.pyg.org/whl/torch-1.12.1%2Bcu116.html").read().decode()
links = re.findall(r'href=[\'\"]?([^\'\" >]+)', html)
print("Found Windows Python 3.8 AMD64 links:")
for l in links:
    if "cp38" in l and "win_amd64" in l:
        print(l)
