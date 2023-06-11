This notebook contains an example of applying reverse image search using SerpAPI ```https://serpapi.com/``` to retrieve a wiki link. 
This is the example image used in the notebook: ![Monalisa -- Leonardo Da Vinchi](https://upload.wikimedia.org/wikipedia/commons/6/6a/Mona_Lisa.jpg)

**In order to run the notebook, please follow the instruction:**

Step 1: Register an account for SerpAPI using this link ```https://serpapi.com/users/sign_up``` and subscribe for the free version which allows to query 200 times a month (I think it's enough for us since we have 1000 times in total for 5 of us).

Step 2: After successful subscription, go to the account page and copy your api key

Step 3: Paste it to the ```API_key``` variable in the notebook and run the notebook



**A few things still need to be optimized or implemented:**

1. It seems that this api only accepts image urls which means we need to first upload it to a file host, while links from some file hosts do not work with the api such as Google Drive and Github.
2. Wiki link retrieved might be a a ```Featured Picture Candidates``` link which is an archive containing all opinions and comments collected for candidate nominations and their nomination results, therefore less informative than other links retrieved by the api. However, it can be improved by remove ```\Wikipedia:Featured_picture_candidates``` from the URL to direct to the desired Wiki page.
3. Implementation of retrieving text from this Wiki link can be implemented for similarities in text level such as historical background, artist intention, etc.




