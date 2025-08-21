train_info_schema = {
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "Train_No": {
        "type": "integer",
        "description": "The unique number of the train."
      },
      "Train_Name": {
        "type": "string",
        "description": "The name of the train."
      },
      "Source_Station_Name": {
        "type": "string",
        "description": "The starting station of the train's journey."
      },
      "Destination_Station_Name": {
        "type": "string",
        "description": "The final destination of the train's journey."
      },
      "days": {
        "type": "string",
        "description": "The day of the week the train operates."
      }
    },
    "required": ["Train_No", "Train_Name", "Source_Station_Name", "Destination_Station_Name", "days"]
  }
}