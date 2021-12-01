## API List ideas



- [ ] getQuota
    -  Params
        - secretKey
    - Returns JSON of how many video mins left as well as how many images left. CAP it to 1 million images and 120 mins of video
- [ ] detectObjectsInImage
    - Params
        - secretKey
        - Image
    - Returns either Image results in a JSON format or a quota finished or No image sent
- [ ] detectObjectsInVideo
    - Params
        - secretKey
        - Video
    - Returns either Video results in a JSON format with timestamps or a quota finished or No image sent