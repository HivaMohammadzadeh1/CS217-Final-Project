# This script creates an S3 bucket 

# Create an S3 bucket (choose a unique bucket name)
aws s3 mb s3://${DCP_BUCKET_NAME} --region ${REGION}

# Create folder for your tarball files
aws s3 mb s3://${DCP_BUCKET_NAME}/${DCP_FOLDER_NAME}/

# Create a folder to keep your logs
aws s3 mb s3://${LOGS_BUCKET_NAME}/${LOGS_FOLDER_NAME}/ --region ${REGION}

# Create a temp file
touch LOGS_FILES_GO_HERE.txt

# Create the folder on S3
aws s3 cp LOGS_FILES_GO_HERE.txt s3://${LOGS_BUCKET_NAME}/${LOGS_FOLDER_NAME}/