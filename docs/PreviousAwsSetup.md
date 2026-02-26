# Setting up Machine for AWS F2 Implementation 

## Table of Contents
* [Accessing F2 Instance](#accessing-f2-instance)
* [AWS Configure](#aws-configure)
    * [New user](#new-user)
* [Create S3 bucket](#create-s3-bucket)
* [Stopping Instance](#stopping-instance)
* [Creating an Alarm](#creating-an-alarm)



## Accessing F2 Instance 
1. You should have received an email with a subject "[CS217] AWS Instance Instructions". Download the private key attached to the email and save it under your `.ssh` folder. Your path to the `.ssh` folder is usually:
    * Linux / Mac : `~/.ssh`
    * Windows: `\Users\$USERNAME\.ssh` or `\user\$USERNAME\.ssh` (replace the $USERNAME with your laptop's username)
2. Sign into the AWS account using the link and information in the email. It will require you to change your password if it's your first time logging in.
3. Once you're logged in, search for the 'ec2' service. If you click on EC2, this will bring you to the screen below. Press 'Instances'.
    <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
        <img src="./images/ec2.png" width="70%">
    </div>
4. Select your instance and go to **Instance State > Start instance**. The designated instance is stated in the email with the AWS login instructions.
    <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
        <img src="./images/ec2-instances.png" width="70%">
    </div>
5. Click the 'Instance ID' and press 'connect'. 
    <div style="padding-left: 0px; padding-top: 0px; text-align: center;">
        <img src="./images/ec2-connect.png" width="50%">
    </div>

    Move to the 'SSH Client' tab and copy the address shown in the boxed field in this picture. This address changes whenever you start the instance. 
    <div style="padding-left: 0px; padding-top: 10px; padding-bottom: 30px; text-align: center;">
        <img src="./images/instance-ip.png" width="50%">
    </div>
6. SSH into your instance
    1. Option 1: VSCode (We recommend this option when you edit code & run)
        1. Install the 'remote-ssh' extension. Press the small button on the lower right and select **Connect to host > Configure SSH Hosts** and choose the first file (this will look something like `\user\$USERNAME\.ssh\config`).
        <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
            <img src="./images/vscode.png" width="50%">
        </div>

        2. Add the following entry to the file and save it. The value for `HostName` is the address you copied in step 5. **This address changes whenever you start the instance. So you will have to update this field whenever you stop and re-start the instance.** The 'IdentityFile' field is the location where you saved the private key file in step 1. 
        ```
        Host aws
        HostName ec2-18-207-233-58.compute-1.amazonaws.com
        IdentityFile "C:\Users\Smriti\.ssh\psmriti.pem"
        User ubuntu
        ```
        3. Go back and press the small button on the lower right you clicked in step 6-1 and select **Connect to host**. The host you've just added will now appear. Selecting the newly added AWS host will connect you to your instance. Once you're connected, you can open folders in your instance using the 'File > Open Folder' feature and run code using the 'Terminal > New Terminal' feature.

    2. Option 2: Terminal (We recommend this option when you want to open a gui to view the emulation reports). The value after the `-i` option is the location of the private key.
    ```
    ssh -i \Users\Smriti\.ssh\psmriti.pem ubuntu@ec2-18-207-233-58.compute-1.amazonaws.com
    ``` 

## AWS Configure
1. Obtain your AWS Access key and secret access key - navigate to IAM from AWS dashboard. Navigate to user > [Your user name]. If a user doesn't exist, check the section [New User](#new-user)
2. Please check the that the following permissions exist 
    <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
            <img src="./images/AwsInstanceSetup_13.png" width="100%">
    </div>
3. create Access key 1 with the the following settings
<div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
            <img src="./images/AwsInstanceSetup_14.png" width="100%">
    </div>
<div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
            <img src="./images/AwsInstanceSetup_15.png" width="100%">
    </div>
4. Yoi can "Set description tag" if you want and proceed with "Create Access Key"
    <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
            <img src="./images/AwsInstanceSetup_16.png" width="100%">
    </div>
5. Make a note of the "Access Key" and "Secret Access key". You can only view this once, so please download the .csv 
    <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
            <img src="./images/AwsInstanceSetup_17.png" width="100%">
    </div>
6. On your instance, configure the aws settings `aws configure`. Set your credentials, region, and output format. If you run the following command, it will ask you for Access Key ID and Secret Access Key, For the region, write 'us-east-1' and for the output write `json`

 ```
 aws configure
 ```
 The result should look like:
 ```
 [centos@ip-172-31-21-2 src]$ aws configure
 AWS Access Key ID [None]: <Your Access Key>
 AWS Secret Access Key [None]: <Your Secret Access Key>
 Default region name [None]: us-east-1
 Default output format [None]: json
 ```

 ### New user
 1. Create a new user and add the user name
 2. `Attach Policies Directly` and add the permissions as give below
    <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
            <img src="./images/AwsInstanceSetup_13.png" width="100%">
        </div>
3. After user is created please go to the steps outlined in [AWS Configure](#aws-configure)

## Create S3 bucket 
S3 Bucket is used to upload and store the generated Amazon FPGA image (AFI). Please follow the steps below to create the S3 bucket
1. Add the following to the bashrc (`~/.bashrc`) and open a new terminal
```
export DCP_BUCKET_NAME=<SUNetID>
export DCP_FOLDER_NAME=cs217_labs
export REGION=us-east-1
export LOGS_FOLDER_NAME=logs_folder
export LOGS_BUCKET_NAME=$DCP_BUCKET_NAME
```
2. Run the following commands. If your chosen DCP_BUCKET_NAME gives you an error in creating the S3 bucket, use any unique ID.
```
# Create an S3 bucket (choose a unique bucket name)
aws s3 mb s3://${DCP_BUCKET_NAME} --region ${REGION}

# Create a folder for your tarball files
aws s3 mb s3://${DCP_BUCKET_NAME}/${DCP_FOLDER_NAME}/

# Create a folder to keep your logs
aws s3 mb s3://${LOGS_BUCKET_NAME}/${LOGS_FOLDER_NAME}/ --region ${REGION}

# Create a temp file
touch LOGS_FILES_GO_HERE.txt

# Create the folder on S3
aws s3 cp LOGS_FILES_GO_HERE.txt s3://${LOGS_BUCKET_NAME}/${LOGS_FOLDER_NAME}/
```

## Stopping Instance
The credits are used based on the number of hours the instance is run. Always turn off the instance once you are done
1. To the stop the instance navigate to the EC2 > Instances. Clock "stop intance" nder the "Instance State' drop down. It will take sometime for the instance to shut down. 
    <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
            <img src="./images/AwsInstanceSetup_18.png" width="100%">
    </div>
    <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
        <img src="./images/AwsInstanceSetup_19.png" width="100%">
    </div>
2. You can also stop it by running the following command on the terminal `sudo shutdown -h now`

## Creating an Alarm 
Please create an alarm for the instance so that it stops once the instance has been unintentionally not turned off. However, please always manually stop the instnace. 
1. Create the alarm and configure the settings as shown 
    <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
            <img src="./images/AwsInstanceSetup_20.png" width="100%">
    </div>
    <div style="padding-left: 0px; padding-bottom: 30px; text-align: center;">
        <img src="./images/AwsInstanceSetup_21.png" width="100%">
    </div>
2. scroll down and select "Create"