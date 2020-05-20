sudo docker image build --rm -t solsol .
sudo docker tag solsol stor.highloadcup.ru/vkcup/marked_snake
sudo docker rmi $(sudo docker images -f "dangling=true" -q)
sudo docker container run --rm --mount type=bind,source=/home/daniil/programming/vk_ads/solsol,target=/tmp/data  stor.highloadcup.ru/vkcup/marked_snake