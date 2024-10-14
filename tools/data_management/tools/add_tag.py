import os

import minio
from minio.commonconfig import Tags
import pymongo
from tqdm import tqdm
from loguru import logger

# oss configuration
host_type = "local"
tag_name = "metadata.experiment_supervised.tag"
tag_value = "tshirt_short_action14_real_zero_center_supervised_v10"

if host_type == "local":
    oss_host = "oss.robotflow.ai"
elif host_type == "hpc":
    oss_host = "oss-hpc.robotflow.ai"
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
oss_port = 443
secure = True
bucket_name = "unifolding"
oss_tag = Tags()
oss_tag[tag_name] = tag_value

# database configuration
db_username = "unifolding"
db_password = "unifolding"
db_host = "192.168.2.223"
db_port = 27017
tag_filter = {tag_name: tag_value}\
    
# -----------------------------------------------------------------------

oss_client = minio.Minio(endpoint=f"{oss_host}:{oss_port}",
                        access_key=access_key,
                        secret_key=secret_key,
                        secure=secure)
buckets = oss_client.list_buckets()
logger.info(f"buckets: {buckets}")
assert bucket_name in map(lambda x: x.name, buckets)

db_connection = pymongo.MongoClient(
    f"mongodb://{db_username}:{db_password}@{db_host}:{db_port}/")

database = db_connection["unifolding"]["logs"]
entries = list(database.find(tag_filter))

for entry in tqdm(entries):
    identifier = entry["identifier"].strip()
    logger.info(f"processing identifier: {identifier}")
    objects = list(oss_client.list_objects(bucket_name, prefix=identifier, recursive=True))    
    if len(objects) == 0:
        logger.warning(f"no object found for identifier {identifier}")
        continue
    for obj in objects:
        logger.info(f"processing object: {obj.object_name}")
        oss_client.set_object_tags(bucket_name, obj.object_name, oss_tag)

del oss_client