# info_measures
Python implementations of information theoretic measures

To install, you first need to build the kdtree copy.

```
python cythonize.py spatial
cd spatial
python setup.py build
cp build/lib.xxx/spatial/ckdtree.xx ../info_measures/spatial/.
```

Then you can install the module with

```
python setup.py develop
```
