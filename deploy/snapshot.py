# deploy/snapshot.py  —  python deploy/snapshot.py /data/memory.db /tmp/snap.db && aws s3 cp /tmp/snap.db s3://my-bucket/snapshots/memory-$(date +%Y%m%dT%H%M%S).db
import os, sqlite3, sys
import sqlite_vec

src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(dst):
    os.remove(dst)
conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
conn.enable_load_extension(True)
sqlite_vec.load(conn)                 # vec0 仮想テーブルを含む DB なので拡張をロードしておく
conn.enable_load_extension(False)
conn.execute("VACUUM INTO ?", (dst,))
conn.close()
