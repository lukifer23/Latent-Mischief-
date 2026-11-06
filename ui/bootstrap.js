"use strict";
let startupFailure="";
function reportStartupFailure(text) {
  startupFailure=text;
  const status=document.getElementById('activityText'),message=document.getElementById('message');
  if(status)status.textContent='Startup error';
  if(message)message.textContent='The lab did not finish loading. Refresh this page after the server reconnects. '+text;
}
window.addEventListener('error',event=>reportStartupFailure(event.message||'A required page resource failed to load.'),true);
window.addEventListener('unhandledrejection',event=>reportStartupFailure(String(event.reason?.message||event.reason||'Unknown startup failure')));
document.addEventListener('DOMContentLoaded',()=>{if(startupFailure)reportStartupFailure(startupFailure);});
