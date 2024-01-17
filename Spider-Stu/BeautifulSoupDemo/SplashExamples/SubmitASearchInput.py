import requests
import StringToFile as fprint
import THKConfig

# Splash服务的地址，注意：如果带有自定义Lua逻辑，此处需要用execute
splash_url = 'http://192.168.146.128:8050/execute'

# 要爬取的目标网页URL
url = THKConfig.decrypt('gKnFKSXrq1kPXvfC/dlkY+GBaWZe3ebmUFQPUP7jjnQ=')

# Splash Examples
lua_script = """

function find_search_input(inputs)
  if #inputs == 1 then
    return inputs[1]
  else
    for _, input in ipairs(inputs) do
      if input.node.attributes.type == "search" then
        return input
      end
    end
  end
end

function find_input(forms)
  local potential = {}

  for _, form in ipairs(forms) do
    local inputs = form.node:querySelectorAll('input:not([type="hidden"])')
    if #inputs ~= 0 then
      local input = find_search_input(inputs)
      if input then
        return form, input
      end

      potential[#potential + 1] = {input=inputs[1], form=form}
    end
  end

  return potential[1].form, potential[1].input
end

function main(splash, args)
  -- find a form and submit "splash" to it
  local function search_for_splash()
    local forms = splash:select_all('form')

    if #forms == 0 then
      error('no search form is found')
    end

    local form, input = find_input(forms)

    if not input then
      error('no search form is found')
    end

    assert(input:send_keys('dog'))
    assert(splash:wait(0))
    assert(form:submit())
  end

  -- main rendering script
  assert(splash:go(args.url))
  assert(splash:wait(1))
  search_for_splash()
  assert(splash:wait(3))

  return splash:png()
end

"""

# 调试上述Lua代码，可浏览器打开http://192.168.146.128:8050
# 在RenderMe中填入目标url和需要调试的Lua脚本
# 可以从日志中拿到界面调试Lua脚本时所使用的Splash请求参数，可以借鉴


# Splash请求参数
params = {
    'url': url,
    'lua_source': lua_script,
}

# 发起Splash请求
response = requests.get(splash_url, params=params)

# 打印渲染后的页面内容
fprint.printToFile(response.text, "using.html")
