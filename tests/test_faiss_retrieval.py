import unittest

import subprocess
import sys


class FaissTests(unittest.TestCase):
    def test_index_flat_ip_matches_brute_force(self):
        code = """import faiss, numpy as np
items=np.array([[1,0],[0,1],[1,1]],dtype='float32'); query=np.array([[.8,.2]],dtype='float32')
faiss.normalize_L2(items); faiss.normalize_L2(query)
index=faiss.IndexFlatIP(2); index.add(items)
actual=index.search(query,3)[1]; expected=np.argsort(-(query@items.T),axis=1)
assert np.array_equal(actual,expected)
"""
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
