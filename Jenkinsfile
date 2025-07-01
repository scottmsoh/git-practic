pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test for PR to develop') {
            when {
                allOf {
                    expression { env.CHANGE_ID != null }
                    expression { env.CHANGE_TARGET == 'develop' }
                }
            }
            steps {
                sh 'python3 ./test/test.py'
            }
        }
    }
}
