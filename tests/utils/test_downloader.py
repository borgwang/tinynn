import tempfile

import tinynn as tn


def test_md5_checksum():
    md5_test_vectors = [
        ("", "d41d8cd98f00b204e9800998ecf8427e"),
        ("a", "0cc175b9c0f1b6a831c399e269772661"),
        ("abc", "900150983cd24fb0d6963f7d28e17f72"),
        ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
        ("abcdefghijklmnopqrstuvwxyz", "c3fcd3d76192e4007dfb496cca67e13b"),
        ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
         "d174ab98d277d9f5a5611c2c9f419d9f")
    ]
    for content, md5_expect in md5_test_vectors:
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(bytes(content, "utf-8"))
            temp.flush()
            assert tn.downloader.md5_checksum(temp.name) == md5_expect
