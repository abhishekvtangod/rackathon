document.getElementById("bn").addEventListener("click", open);

function toggleNav(){
    navSize = document.getElementById("iframeHolder").style.width;
    if (navSize == 350) {
        return close();
    }
    return open();
}
function open() {
        window.alert("Hello")
        document.getElementById("iframeHolder").style.width = "350px";
        document.getElementById("iframeHolder").style.height = "430px";
        document.getElementById("iframeHolder").src="https://console.dialogflow.com/api-client/demo/embedded/c42f8df7-e888-48f8-b3fd-320c401538d7";
        
}
function close() {
         document.getElementById("iframeHolder").style.width = "0";
          document.getElementById("iframeHolder").style.width = "0";
       
}