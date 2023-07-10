import boto3

dynamodb = boto3.resource('dynamodb')

table = dynamodb.create_table(
    TableName='LangChainSessionTable',
    KeySchema=[
        {
            'AttributeName': 'SessionId',
            'KeyType': 'HASH',
        },
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'SessionId',
            'AttributeType': 'S',
        },
    ],
    BillingMode='PAY_PER_REQUEST',
)

table.meta.client.get_waiter('table_exists').wait(TableName='LangChainSessionTable')

print(table.item_count)
