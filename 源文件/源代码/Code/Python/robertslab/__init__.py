import sys
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)
__all__ = ["avro","cellio","md","ndarray","peaks","pbuf","sfile"]

# Add robertslab from the protobuf and avro zip files.
for p in sys.path:
    if p.endswith("robertslab-protobuf-python.zip"):
        __path__.append(p+"/robertslab")
    if p.endswith("robertslab-avro-python.zip"):
        __path__.append(p+"/robertslab")
