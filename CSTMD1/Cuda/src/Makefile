CC = nvcc
TARGET = multi
LFLAGS = -lcublas -lcurand -lcuda
OBJS = main.o Cstmd1Sim.o

$(TARGET): $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o $(TARGET)

main.o: main.cu
	$(CC) -c main.cu

Cstmd1Sim.o: Cstmd1Sim.cu
	$(CC) -c Cstmd1Sim.cu

clean:
	rm -rf *.o $(TARGET)
