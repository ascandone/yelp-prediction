mkdir -p ./data/temp

cd ./data

# download and unzip the json dataset
curl -L -o ./temp/json.zip https://business.yelp.com/external-assets/files/Yelp-JSON.zip -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" 
unzip -o ./temp/json.zip -d ./temp/json
mkdir -p ./dataset-json
tar -xvf "./temp/json/Yelp JSON/yelp_dataset.tar" -C ./dataset-json

# download and unzip the photos dataset
curl -L -o ./temp/photos.zip https://business.yelp.com/external-assets/files/Yelp-Photos.zip -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" 
unzip -o ./temp/photos.zip -d ./temp/photos
mkdir -p ./dataset-photos
tar -xf "./temp/photos/Yelp Photos/yelp_photos.tar" -C ./dataset-photos

# cleanup temp files
rm -rf ./temp
