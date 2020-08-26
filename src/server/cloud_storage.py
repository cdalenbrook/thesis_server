from google.cloud import storage
BUCKETNAME = 'yuki-b8063.appspot.com'


def upload_image(bucket: storage.Bucket, path: str, image: str) -> str:
    blob = bucket.blob(path)
    contents = blob.upload_from_string(
        image, content_type='image/png')
    print(contents)
    return f'http://storage.googleapis.com/{BUCKETNAME}/{path}'
