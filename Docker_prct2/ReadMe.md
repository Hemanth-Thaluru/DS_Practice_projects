## SImple Project for learning `Docker` and `FastAPI` 

### Steps followed
1. Intially built docker image with help of Dockerfile
2. Connected local volumes to container for hot reload
3. Moved to conatiner based vscode developement(opened the src file in container in local vscode) to get package suggestions
4. With help of docker-compose file run multiple images at the same time
5. With help of debugpy package - Debugging made possible on fly
 

### What I did & how I did

1. For image building 
```
docker run --name fastapi-container -p 80:80 -d fastapi-image
```
2. For Hot reloading 
```
docker run --name fastapi-container -p 80:80 -d -v $(pwd):/code fastapi-image
```
3. Open vs code in container with the help(vs extension)- Attach to running container

4. Multiple images at once using docker-compose file
```
docker-compose up --build -d 

```
5. Debugging on fly
```
import debugpy 

```