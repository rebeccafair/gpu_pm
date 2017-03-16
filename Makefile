EXEC=patternTest
SRC_DIR=src
OBJ_DIR=obj
INC_DIR=include
LIB_DIR=lib

CC=g++
CFLAGS=-std=c++0x -I./$(INC_DIR)

SOURCES:=eventReader.cpp
SOURCES+=patternReader.cpp
SOURCES+=matchPatterns.cpp
SOURCES+=main.cpp

OBJS = $(addprefix $(OBJ_DIR)/, $(SOURCES:.cpp=.o))

vpath %.cpp $(SRC_DIR)
vpath %.h $(SRC_DIR)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o
	rm -f $(EXEC)
