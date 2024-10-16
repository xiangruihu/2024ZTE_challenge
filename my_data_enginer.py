import dashscope

api_key = ''
from http import HTTPStatus
from dashscope import Application
from dashscope import Generation
dashscope.api_key = ''

from http import HTTPStatus
from dashscope import Application


def call_agent_app():
    response = Application.call(app_id='',
                                # prompt='如何做炒西红柿鸡蛋？',

                                api_key=api_key,

                                )

    if response.status_code != HTTPStatus.OK:
        print('request_id=%s, code=%s, message=%s\n' % (response.request_id, response.status_code, response.message))
    else:
        print('request_id=%s\n output=%s\n usage=%s\n' % (response.request_id, response.output, response.usage))


if __name__ == '__main__':
    call_agent_app()

# if __name__ == '__main__':
#     call_with_stream()
#     # call_agent_app()