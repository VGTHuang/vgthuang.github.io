(function(t){function o(o){for(var n,i,s=o[0],c=o[1],u=o[2],l=0,d=[];l<s.length;l++)i=s[l],Object.prototype.hasOwnProperty.call(a,i)&&a[i]&&d.push(a[i][0]),a[i]=0;for(n in c)Object.prototype.hasOwnProperty.call(c,n)&&(t[n]=c[n]);h&&h(o);while(d.length)d.shift()();return r.push.apply(r,u||[]),e()}function e(){for(var t,o=0;o<r.length;o++){for(var e=r[o],n=!0,i=1;i<e.length;i++){var s=e[i];0!==a[s]&&(n=!1)}n&&(r.splice(o--,1),t=c(c.s=e[0]))}return t}var n={},i={app:0},a={app:0},r=[];function s(t){return c.p+"js/"+({about:"about"}[t]||t)+"."+{about:"0b6df648"}[t]+".js"}function c(o){if(n[o])return n[o].exports;var e=n[o]={i:o,l:!1,exports:{}};return t[o].call(e.exports,e,e.exports,c),e.l=!0,e.exports}c.e=function(t){var o=[],e={about:1};i[t]?o.push(i[t]):0!==i[t]&&e[t]&&o.push(i[t]=new Promise((function(o,e){for(var n="css/"+({about:"about"}[t]||t)+"."+{about:"4a1853a2"}[t]+".css",a=c.p+n,r=document.getElementsByTagName("link"),s=0;s<r.length;s++){var u=r[s],l=u.getAttribute("data-href")||u.getAttribute("href");if("stylesheet"===u.rel&&(l===n||l===a))return o()}var d=document.getElementsByTagName("style");for(s=0;s<d.length;s++){u=d[s],l=u.getAttribute("data-href");if(l===n||l===a)return o()}var h=document.createElement("link");h.rel="stylesheet",h.type="text/css",h.onload=o,h.onerror=function(o){var n=o&&o.target&&o.target.src||a,r=new Error("Loading CSS chunk "+t+" failed.\n("+n+")");r.code="CSS_CHUNK_LOAD_FAILED",r.request=n,delete i[t],h.parentNode.removeChild(h),e(r)},h.href=a;var f=document.getElementsByTagName("head")[0];f.appendChild(h)})).then((function(){i[t]=0})));var n=a[t];if(0!==n)if(n)o.push(n[2]);else{var r=new Promise((function(o,e){n=a[t]=[o,e]}));o.push(n[2]=r);var u,l=document.createElement("script");l.charset="utf-8",l.timeout=120,c.nc&&l.setAttribute("nonce",c.nc),l.src=s(t);var d=new Error;u=function(o){l.onerror=l.onload=null,clearTimeout(h);var e=a[t];if(0!==e){if(e){var n=o&&("load"===o.type?"missing":o.type),i=o&&o.target&&o.target.src;d.message="Loading chunk "+t+" failed.\n("+n+": "+i+")",d.name="ChunkLoadError",d.type=n,d.request=i,e[1](d)}a[t]=void 0}};var h=setTimeout((function(){u({type:"timeout",target:l})}),12e4);l.onerror=l.onload=u,document.head.appendChild(l)}return Promise.all(o)},c.m=t,c.c=n,c.d=function(t,o,e){c.o(t,o)||Object.defineProperty(t,o,{enumerable:!0,get:e})},c.r=function(t){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},c.t=function(t,o){if(1&o&&(t=c(t)),8&o)return t;if(4&o&&"object"===typeof t&&t&&t.__esModule)return t;var e=Object.create(null);if(c.r(e),Object.defineProperty(e,"default",{enumerable:!0,value:t}),2&o&&"string"!=typeof t)for(var n in t)c.d(e,n,function(o){return t[o]}.bind(null,n));return e},c.n=function(t){var o=t&&t.__esModule?function(){return t["default"]}:function(){return t};return c.d(o,"a",o),o},c.o=function(t,o){return Object.prototype.hasOwnProperty.call(t,o)},c.p="/sq-3d-framework/",c.oe=function(t){throw console.error(t),t};var u=window["webpackJsonp"]=window["webpackJsonp"]||[],l=u.push.bind(u);u.push=o,u=u.slice();for(var d=0;d<u.length;d++)o(u[d]);var h=l;r.push([0,"chunk-vendors"]),e()})({0:function(t,o,e){t.exports=e("cd49")},"034f":function(t,o,e){"use strict";e("85ec")},"102f":function(t,o,e){},"61bd":function(t,o,e){"use strict";e("d4d6")},"6cba":function(t,o,e){"use strict";e("102f")},"85ec":function(t,o,e){},a24a:function(t,o,e){"use strict";e("aa74")},aa74:function(t,o,e){},cd49:function(t,o,e){"use strict";e.r(o);e("e260"),e("e6cf"),e("cca6"),e("a79d");var n=e("2b0e"),i=function(){var t=this,o=t.$createElement,e=t._self._c||o;return e("div",{attrs:{id:"app"}},[e("div",{attrs:{id:"nav"}},[e("router-link",{attrs:{to:"/"}},[t._v("Home")]),t._v(" | "),e("router-link",{attrs:{to:"/about"}},[t._v("About")]),t._v(" | "),e("router-link",{attrs:{to:"/ThreeSceneTest"}},[t._v("Scene")])],1),e("router-view",{staticClass:"router-view"})],1)},a=[],r=(e("034f"),e("2877")),s={},c=Object(r["a"])(s,i,a,!1,null,null,null),u=c.exports,l=e("2f62"),d=(e("d3b7"),e("3ca3"),e("ddb0"),e("8c4f")),h=function(){var t=this,o=t.$createElement,e=t._self._c||o;return e("div",{staticClass:"container-all"},[e("div",{staticClass:"stats-panel"},[e("p",[t._v(" "+t._s(t.$store.state.mouseX)+" ")])]),e("div",{staticClass:"bg"}),e("div",{staticClass:"dynamic-container"},[e("card-group",{ref:"card-group"})],1)])},f=[],p=function(){var t=this,o=t.$createElement,e=t._self._c||o;return e("div",{staticClass:"container-3d",class:{"size-100":t.isCenter}},[e("div",{staticClass:"abs-elem card-group-container size-100 container-3d",style:{transform:t.cardGroupPosition.asCSSPosition()}},[e("div",{ref:"cards-container",staticClass:"container-3d flex-center size-100",class:{rotate:t.isRotating}},[t._l(t.cardGroupGeomControl.lids,(function(t,o){return e("card",{key:o,class:t.classes,attrs:{CardGeomControl:t}})})),t._l(t.cardGroupGeomControl.cards,(function(t,o){return e("card",{key:2+o,class:t.classes,attrs:{CardGeomControl:t}})}))],2)]),e("div",{staticClass:"stats-panel"},[e("p",[e("input",{directives:[{name:"model",rawName:"v-model",value:t.cardGroupPosition.x,expression:"cardGroupPosition.x"}],attrs:{type:"range",min:"-300",max:"300",step:"1"},domProps:{value:t.cardGroupPosition.x},on:{__r:function(o){return t.$set(t.cardGroupPosition,"x",o.target.value)}}}),e("input",{directives:[{name:"model",rawName:"v-model",value:t.cardGroupPosition.y,expression:"cardGroupPosition.y"}],attrs:{type:"range",min:"-300",max:"300",step:"1"},domProps:{value:t.cardGroupPosition.y},on:{__r:function(o){return t.$set(t.cardGroupPosition,"y",o.target.value)}}})]),e("p",[e("button",{on:{click:t.startRotate}},[t._v("start rotate")]),e("button",{on:{click:t.endRotate}},[t._v("end rotate")])]),e("p",[e("input",{directives:[{name:"model",rawName:"v-model",value:t.cubeNetAngle,expression:"cubeNetAngle"}],attrs:{type:"range",min:"0",max:"90",step:"1"},domProps:{value:t.cubeNetAngle},on:{input:t.onAngleChange,__r:function(o){t.cubeNetAngle=o.target.value}}}),e("button",{on:{click:t.toCube}},[t._v("to cube")]),e("button",{on:{click:t.toCubeNet1_half}},[t._v("to net (half expanded)")]),e("button",{on:{click:t.toCubeNet1_full}},[t._v("to net (fully expanded)")]),t._v(t._s(t.$store.state.animationObjectCount)+" ")]),e("p",[e("button",{on:{click:t.moveToCenter}},[t._v("group move to center")]),e("button",{on:{click:t.moveToOriginalPosition}},[t._v("group move to original pos")])]),e("p",[e("button",{on:{click:t.toFullDetail}},[t._v("to full detail")]),e("button",{on:{click:t.cancelFullDetail}},[t._v("back to cube")])]),e("p",[e("button",{on:{click:t.toCarousel}},[t._v("to carousel")]),e("button",{on:{click:t.fromCarousel}},[t._v("back to cube")])]),e("p",[e("button",{on:{click:t.toList}},[t._v("to list")]),e("button",{on:{click:t.fromList}},[t._v("back to cube")])])])])},m=[],v=(e("159b"),e("22b5")),b=function(){var t=this,o=t.$createElement,e=t._self._c||o;return e("div",{directives:[{name:"show",rawName:"v-show",value:t.CardGeomControl.visibility,expression:"CardGeomControl.visibility"}],staticClass:"abs-elem card",style:{height:t.CardGeomControl.height+"px",width:t.CardGeomControl.width+"px","border-radius":t.CardGeomControl.borderRadius+"px","transform-origin":t.CardGeomControl.origin.asCSSOrigin(),transform:t.CardGeomControl.position.asCSSPosition()+" "+t.CardGeomControl.rotation.asCSSRotation()}},[e("div",{staticClass:"card__front"},[t._v(" front ")]),e("div",{staticClass:"card__back"},[t._v(" back ")])])},C=[],g=e("d4ec"),G=e("bee2"),y=e("ade3"),O=(e("99af"),function(){function t(){var o=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0,e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0;Object(g["a"])(this,t),Object(y["a"])(this,"x",void 0),Object(y["a"])(this,"y",void 0),this.x=o,this.y=e}return Object(G["a"])(t,[{key:"set",value:function(){var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0,o=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0;this.x=t,this.y=o}},{key:"copy",value:function(){return new t(this.x,this.y)}},{key:"asarray",value:function(){return[this.x,this.y]}},{key:"asCSSPosition",value:function(){return"translateX(".concat(this.x,"px) translateY(").concat(this.y,"px)")}},{key:"asCSSOrigin",value:function(){return"".concat(this.x,"px ").concat(this.y,"px")}}]),t}()),w=function(){function t(){var o=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0,e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0,n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:0;Object(g["a"])(this,t),Object(y["a"])(this,"x",void 0),Object(y["a"])(this,"y",void 0),Object(y["a"])(this,"z",void 0),this.x=o,this.y=e,this.z=n}return Object(G["a"])(t,[{key:"set",value:function(){var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0,o=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0,e=arguments.length>2&&void 0!==arguments[2]?arguments[2]:0;this.x=t,this.y=o,this.z=e}},{key:"copy",value:function(){return new t(this.x,this.y,this.z)}},{key:"asarray",value:function(){return[this.x,this.y,this.z]}},{key:"sum",value:function(){return this.x+this.y+this.z}},{key:"asCSSPosition",value:function(){return"translateX(".concat(this.x,"px) translateY(").concat(this.y,"px) translateZ(").concat(this.z,"px)")}},{key:"asCSSRotation",value:function(){return"rotateX(".concat(this.x,"deg) rotateY(").concat(this.y,"deg) rotateZ(").concat(this.z,"deg)")}}]),t}(),_=Object(G["a"])((function t(o,e,n,i,a){Object(g["a"])(this,t),Object(y["a"])(this,"position",void 0),Object(y["a"])(this,"rotation",void 0),Object(y["a"])(this,"origin",void 0),Object(y["a"])(this,"height",void 0),Object(y["a"])(this,"width",void 0),Object(y["a"])(this,"borderRadius",void 0),Object(y["a"])(this,"visibility",void 0),Object(y["a"])(this,"classes",void 0),this.position=o,this.rotation=e,this.origin=n,this.height=i,this.width=a,this.borderRadius=0,this.visibility=!0,this.classes=""})),x=Object(G["a"])((function t(o){Object(g["a"])(this,t),Object(y["a"])(this,"lids",void 0),Object(y["a"])(this,"cards",void 0),this.lids=[],this.cards=[];for(var e=0;e<2;e++)this.lids.push(new _(new w,new w,new O,o,o));for(var n=0;n<4;n++)this.cards.push(new _(new w,new w,new O,o,o))}));function j(t,o,e,n){var i=Math.sin(e*Math.PI/180),a=Math.cos(e*Math.PI/180),r=n*(e/90);o.cards[0].position.set(-t-r,0,0),o.cards[0].rotation.set(-0,90-e,0),o.cards[0].origin.set(t,0),o.cards[1].position.set(0,0,0),o.cards[1].rotation.set(-0,0,0),o.cards[1].origin.set(0,0),o.cards[2].position.set(t+r,0,0),o.cards[2].rotation.set(-0,-90+e,0),o.cards[2].origin.set(0,0),o.cards[3].position.set(t+r+i*(t+r),0,a*(t+r)),o.cards[3].rotation.set(-0,2*e-180,0),o.cards[3].origin.set(0,0),o.lids[0].position.set(0,-t-r,0),o.lids[0].origin.set(0,t),o.lids[0].rotation.set(-90+e,0,0),o.lids[1].position.set(0,t+r,0),o.lids[1].rotation.set(90-e,0,0),o.lids[1].origin.set(0,0)}var k=n["a"].extend({name:"Card",props:{CardGeomControl:{type:_}},computed:{}}),S=k,P=(e("61bd"),Object(r["a"])(S,b,C,!1,null,"72914d05",null)),E=P.exports;n["a"].use(l["a"]);var A,N=new l["a"].Store({state:{mouseX:0,mouseY:0,animationObjectCount:0},mutations:{onMouseMove:function(t,o){t.mouseX=o.clientX,t.mouseY=o.clientY},addAnimationObject:function(t){t.animationObjectCount++},removeAnimationObject:function(t){t.animationObjectCount--}},actions:{addTweenAnimObject:function(t){t.state.animationObjectCount<=0&&z(),t.commit("addAnimationObject")},removeTweenAnimObject:function(t){t.state.animationObjectCount<=0||(1==t.state.animationObjectCount&&L(),t.commit("removeAnimationObject"))}}}),T=N,M=-1;function R(){v["c"](),console.log("update"),M>=0&&(M=window.requestAnimationFrame(R))}function z(){M<0&&(M=window.requestAnimationFrame(R))}function L(){M>=0&&(window.cancelAnimationFrame(M),M=-1)}function F(t,o,e,n){var i=arguments.length>4&&void 0!==arguments[4]?arguments[4]:1e3,a=arguments.length>5&&void 0!==arguments[5]?arguments[5]:v["a"].Quadratic.InOut,r={a:t};T.dispatch("addTweenAnimObject"),new v["b"](r).to({a:o},i).easing(a).onUpdate((function(){return e(r.a)})).onComplete((function(){T.dispatch("removeTweenAnimObject"),n(r.a)})).start()}function $(t,o,e,n){var i=arguments.length>4&&void 0!==arguments[4]?arguments[4]:1e3,a=arguments.length>5&&void 0!==arguments[5]?arguments[5]:v["a"].Quadratic.InOut,r={x:t.x,y:t.y},s={x:o.x,y:o.y};T.dispatch("addTweenAnimObject"),new v["b"](r).to(s,i).easing(a).onUpdate((function(){return e(r.x,r.y)})).onComplete((function(){T.dispatch("removeTweenAnimObject"),n(r.x,r.y)})).start()}(function(t){t[t["Cube"]=0]="Cube",t[t["CubeRotate"]=1]="CubeRotate",t[t["CubeNet1_half"]=2]="CubeNet1_half",t[t["CubeNet1_full"]=3]="CubeNet1_full",t[t["Carousel"]=4]="Carousel",t[t["List"]=5]="List",t[t["Shuffle"]=6]="Shuffle"})(A||(A={}));var I=n["a"].extend({name:"CardGroup",components:{Card:E},data:function(){return{cardGroupPosition:new O(0,0),cardGroupGeomControl:new x(160),defaultCubeSize:160,cubeNetAngle:0,isRotating:!1,onRotateEndFunc:function(){},phase:A.Cube,isCenter:!1,originalCardGroupPosition:new O(0,0)}},created:function(){j(this.defaultCubeSize,this.cardGroupGeomControl,0,10)},methods:{onAngleChange:function(){j(this.defaultCubeSize,this.cardGroupGeomControl,this.cubeNetAngle-0,20)},startRotate:function(){this.isRotating=!0,this.phase=A.CubeRotate},cancelRotatingState:function(){this.isRotating=!1,this.phase=A.Cube,this.onRotateEndFunc instanceof Function&&this.onRotateEndFunc();var t=this.$refs["cards-container"];t instanceof Element&&t.removeEventListener("animationiteration",this.cancelRotatingState)},endRotate:function(t){this.onRotateEndFunc=t;var o=this.$refs["cards-container"];o instanceof Element&&o.addEventListener("animationiteration",this.cancelRotatingState)},toCube:function(){var t=this;F(this.cubeNetAngle-0,0,(function(o){t.cubeNetAngle=o,t.onAngleChange()}),(function(){t.phase=A.Cube}),1e3,v["a"].Exponential.InOut)},toCubeNet1_half:function(){var t=this;F(this.cubeNetAngle-0,60,(function(o){t.cubeNetAngle=o,t.onAngleChange()}),(function(){t.phase=A.CubeNet1_half}),1e3,v["a"].Cubic.InOut)},toCubeNet1_full:function(){var t=this;F(this.cubeNetAngle-0,90,(function(o){t.cubeNetAngle=o,t.onAngleChange()}),(function(){t.phase=A.CubeNet1_full}),1e3,v["a"].Bounce.Out)},moveToCenter:function(){this.originalCardGroupPosition=this.cardGroupPosition.copy();var t=this;$(this.originalCardGroupPosition,new O(0,0),(function(o,e){t.cardGroupPosition.x=o,t.cardGroupPosition.y=e}),(function(){t.isCenter=!0}),1e3,v["a"].Cubic.InOut)},moveToOriginalPosition:function(){if(!1!==this.isCenter){var t=this;$(this.cardGroupPosition,this.originalCardGroupPosition,(function(o,e){t.cardGroupPosition.x=o,t.cardGroupPosition.y=e}),(function(){t.isCenter=!1}),1e3,v["a"].Cubic.InOut)}},toFullDetail:function(){this.cardGroupGeomControl.lids.forEach((function(t){t.classes="smooth-transition"})),this.cardGroupGeomControl.lids[0].classes+=" hide-top",this.cardGroupGeomControl.lids[1].classes+=" hide-bottom",this.cardGroupGeomControl.cards.forEach((function(t){t.classes="smooth-transition"})),this.cardGroupGeomControl.cards[0].classes+=" hide-left",this.cardGroupGeomControl.cards[1].classes+=" full-detail",this.cardGroupGeomControl.cards[2].classes+=" hide-right",this.cardGroupGeomControl.cards[3].classes+=" left-bar"},cancelFullDetail:function(){var t=this;this.cardGroupGeomControl.lids.forEach((function(t){t.classes="smooth-transition"})),this.cardGroupGeomControl.cards.forEach((function(t){t.classes="smooth-transition"})),setTimeout((function(){t.cardGroupGeomControl.lids.forEach((function(t){t.classes=""})),t.cardGroupGeomControl.cards.forEach((function(t){t.classes=""}))}),1010)},toCarousel:function(){var t=this;this.cardGroupGeomControl.lids.forEach((function(t){t.visibility=!1})),F(0,1,(function(o){t.cardGroupGeomControl.cards.forEach((function(e){e.height=t.defaultCubeSize*(1+1*o),e.width=t.defaultCubeSize*(1+.4*o)})),j(t.cardGroupGeomControl.cards[0].width,t.cardGroupGeomControl,110*o,0)}),(function(){t.phase=A.Carousel}),1e3,v["a"].Cubic.InOut)},fromCarousel:function(){var t=this;F(1,0,(function(o){t.cardGroupGeomControl.cards.forEach((function(e){e.height=t.defaultCubeSize*(1+1*o),e.width=t.defaultCubeSize*(1+.4*o)})),j(t.cardGroupGeomControl.cards[0].width,t.cardGroupGeomControl,110*o,0)}),(function(){t.cardGroupGeomControl.lids.forEach((function(t){t.visibility=!0})),t.phase=A.Cube}),1e3,v["a"].Cubic.InOut)},toList:function(){this.cardGroupGeomControl.lids.forEach((function(t){t.classes="smooth-transition card-as-list"})),this.cardGroupGeomControl.cards.forEach((function(t){t.classes="smooth-transition card-as-list"})),this.cardGroupGeomControl.lids.forEach((function(t,o){t.rotation.set(30*Math.random()-15,-20*(o%2-.5)+30*Math.random()-15,30*Math.random()-15)})),this.cardGroupGeomControl.cards.forEach((function(t,o){t.rotation.set(30*Math.random()-15,-20*(o%2-.5)+30*Math.random()-15,30*Math.random()-15)}));var t=(this.defaultCubeSize+20)/2;this.cardGroupGeomControl.lids[0].position.set(-t,2*-t,-10),this.cardGroupGeomControl.lids[1].position.set(t,2*-t,-10),this.cardGroupGeomControl.cards[0].position.set(-t,0,-10),this.cardGroupGeomControl.cards[1].position.set(t,0,-10),this.cardGroupGeomControl.cards[2].position.set(-t,2*t,-10),this.cardGroupGeomControl.cards[3].position.set(t,2*t,-10)},fromList:function(){var t=this;j(this.defaultCubeSize,this.cardGroupGeomControl,0,0),setTimeout((function(){t.cardGroupGeomControl.lids.forEach((function(t){t.classes=""})),t.cardGroupGeomControl.cards.forEach((function(t){t.classes=""}))}),1010)}}}),D=I,X=(e("a24a"),Object(r["a"])(D,p,m,!1,null,"d190e6f8",null)),q=X.exports,Y={name:"Home",components:{CardGroup:q},data:function(){return{}},mounted:function(){window.addEventListener("mousemove",this.onMouseMove)},methods:{onMouseMove:function(t){this.$store.commit("onMouseMove",t)}},beforeDestroy:function(){window.removeEventListener("mousemove",this.onMouseMove)}},B=Y,H=(e("6cba"),Object(r["a"])(B,h,f,!1,null,"7466ff26",null)),U=H.exports;n["a"].use(d["a"]);var J=[{path:"/",name:"Home",component:U},{path:"/about",name:"About",component:function(){return e.e("about").then(e.bind(null,"f820"))}},{path:"/ThreeSceneTest",name:"ThreeSceneTest",component:function(){return e.e("about").then(e.bind(null,"1da8"))}}],Q=new d["a"]({mode:"history",base:"/sq-3d-framework/",routes:J}),Z=Q;n["a"].config.productionTip=!1,n["a"].use(l["a"]),new n["a"]({router:Z,store:T,render:function(t){return t(u)}}).$mount("#app")},d4d6:function(t,o,e){}});
//# sourceMappingURL=app.ec5843fd.js.map