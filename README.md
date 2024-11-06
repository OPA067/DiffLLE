drwxr-xr-x 1 root root       2696 11月  6 03:37 config
drwxr-xr-x 1 root root      40441 11月  6 03:38 core
drwxr-xr-x 1 root root 1401467135 11月  6 06:32 data
drwxr-xr-x 1 root root     279529 11月  6 03:38 model
-rw-r--r-- 1 root root        155 11月  6 02:45 README.md
-rw-r--r-- 1 root root       8401 11月  6 04:00 sr.py
drwxr-xr-x 1 root root      46957 11月  6 03:42 utils

```python training
python sr.py -p train -c config/light_LOL_v1.json  
```

```python testing
python sr.py -p val -c config/light_LOL_v1.json
```

