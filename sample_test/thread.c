#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define MAX_SEQ_NUMBER 	(100)

#define MAX_THREAD_NUMBER (30)

#define MAX_PRINT_NUM	(10)

struct thread_param {
	int seq;
	pthread_mutex_t *my_thread_mutex;
	pthread_mutex_t *next_thread_mutex;
};



void *thread_start_routin(void *arg)
{
	int i;
	int num = rand();
	struct thread_param *param =  (struct thread_param*)arg;

	for (i = 0; i< num % MAX_PRINT_NUM;i++) {
		printf("dct %d %d %d\n", param->seq, (num %10), i);
	}
	if (param->my_thread_mutex != NULL ) {
		pthread_mutex_lock(param->my_thread_mutex);
	}
	num = rand();
	for (i = 0; i< num % MAX_PRINT_NUM;i++) {
//		printf("mem write %d %d %d\n", param->seq, (num %10), i);
	}
//	printf("end %d\n", param->seq);
	if (param->next_thread_mutex != NULL ) {
		pthread_mutex_unlock(param->next_thread_mutex);
	}
	
	free(arg);

	pthread_exit(NULL);
}


int main()
{
	int seq_number = 0;
	int ret;
	int thread_number = 0;
	int end_thread_number = 0;

	pthread_attr_t attr;

	pthread_mutex_t *next_thread_mutex = NULL;
	pthread_mutex_t *current_thread_mutex = NULL;
	pthread_t thread[MAX_SEQ_NUMBER];

	for(seq_number=0;seq_number < MAX_SEQ_NUMBER;seq_number++) {
		current_thread_mutex = next_thread_mutex;
		int ret = pthread_attr_init(&attr);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
		struct thread_param * param = (struct thread_param*)malloc(sizeof(struct thread_param));
		param->seq = seq_number; 
		param->my_thread_mutex = current_thread_mutex;

		if (seq_number != (MAX_SEQ_NUMBER -1)) {
			next_thread_mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
			if (next_thread_mutex == NULL) {
				printf("%d\n", __LINE__);
				return -1;
			}
			ret = pthread_mutex_init(next_thread_mutex,NULL);
			if (ret != 0) {
				printf("%d\n", __LINE__);
				return -1;
			}

			ret = pthread_mutex_lock(next_thread_mutex);
			if (ret != 0) {
				printf("%d\n", __LINE__);
				return -1;
			}
			param->next_thread_mutex = next_thread_mutex;

		} else {
		}
		
		ret = pthread_create(&thread[seq_number], &attr, &thread_start_routin, (void*)param);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
		thread_number++;
		if (thread_number == MAX_THREAD_NUMBER) {
//			printf("join %d %d\n", thread_number, end_thread_number);
			pthread_join(thread[end_thread_number], NULL);
			end_thread_number++;
			thread_number--;
			if (end_thread_number == MAX_SEQ_NUMBER) {
				break;
			}

		}

	}
	if (end_thread_number != MAX_SEQ_NUMBER) {
		for(;;) {
//			printf("join %d %d\n", thread_number, end_thread_number);
			pthread_join(thread[end_thread_number], NULL);
			end_thread_number++;
			thread_number--;
			if (end_thread_number == MAX_SEQ_NUMBER) {
				break;
			}
		}
	}
	printf("main end %d\n", thread_number);
	return 0;


}