EXEC=patternTest
OBJ_DIR=obj
INC_DIR=include
LIB_DIR=lib

CC=g++
CFLAGS=-I./$(INC_DIR)

DEPS=  #header files

OBJ_FILES:=helloWorld.cpp
OBJ_FILES+=patternTest.cpp

$(OBJ_DIR)/%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

$(EXEC): $(OBJ_FILES)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(OBJ_DIR)/*.o
	rm -f $(EXEC)
